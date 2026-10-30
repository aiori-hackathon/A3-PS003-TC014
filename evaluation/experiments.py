"""
Master Experiment Runner for AIORI Hackathon 3 PS-003.
Orchestrates:
1. Four-Paradigm Benchmark across 3 Random Seeds (42, 123, 999):
   - Experiment A: Persistence Baseline
   - Experiment B: Centralized STGCN
   - Experiment C: Standard FedAvg (IID & Non-IID)
   - Experiment D: Proposed BEFA (IID & Non-IID)
2. Ablation Studies:
   - Event Triggering: Enabled vs Disabled
   - Compression: Uncompressed vs 8-bit Quantized
   - Budget Sensitivity: 2.0 MB, 3.5 MB, 5.0 MB
Exports comprehensive CSV results to results/metrics/master_experiment_results.csv.
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.stgcn import CompactSTGCN
from data.dataset_loader import PEMS04DataManager, generate_sliding_windows
from clients.partitioner import FederatedPartitioner
from clients.client import VirtualClient
from server.coordinator import FederatedCoordinator
from server.befa_coordinator import BEFACoordinator
from training.centralized import CentralizedTrainer
from evaluation.metrics import evaluate_persistence_baseline, evaluate_forecast_metrics


def set_seed(seed: int):
    """Enforces reproducible deterministic execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ExperimentHarness:
    """
    Executes and logs multi-seed and ablation experiments.
    """
    def __init__(
        self,
        batch_size: int = 128,
        rounds: int = 3,
        quick: bool = False,
        results_dir: str = "results/metrics"
    ):
        self.batch_size = batch_size
        self.rounds = rounds
        self.quick = quick
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Prepare base dataset manager
        self.data_mgr = PEMS04DataManager(
            npz_path="pems04.npz",
            csv_path="data/adjacency/PEMS04.csv",
            in_steps=12,
            out_steps=3,
            batch_size=batch_size
        ).prepare()

        self.model_cfg = {
            "num_features": 3,
            "in_steps": 12,
            "out_steps": 3,
            "hidden_channels": 32,
            "k_order": 2,
            "kt": 3,
            "dropout": 0.1
        }

        self.all_records: List[Dict[str, Any]] = []

    def run_persistence_baseline(self) -> Dict[str, float]:
        """Runs Experiment A: Historical Persistence Baseline."""
        raw_test_x, raw_test_y = generate_sliding_windows(
            self.data_mgr.test_data_raw,
            in_steps=self.data_mgr.in_steps,
            out_steps=self.data_mgr.out_steps
        )
        if self.quick:
            raw_test_x = raw_test_x[:1000]
            raw_test_y = raw_test_y[:1000]

        return evaluate_persistence_baseline(raw_test_x, raw_test_y)

    def run_centralized_trial(self, seed: int) -> Dict[str, Any]:
        """Runs Experiment B: Centralized STGCN Training."""
        set_seed(seed)
        model = CompactSTGCN(**self.model_cfg)
        trainer = CentralizedTrainer(
            model=model,
            data_manager=self.data_mgr,
            learning_rate=1e-3,
            weight_decay=1e-4,
            checkpoint_dir="results/checkpoints",
            log_dir="results/logs"
        )
        epochs = 2 if self.quick else 3
        summary = trainer.fit(epochs=epochs, checkpoint_name=f"centralized_seed{seed}.pt")
        return summary

    def _setup_clients(self, distribution: str, seed: int) -> List[VirtualClient]:
        """Creates virtual clients under specified distribution and seed."""
        partitioner = FederatedPartitioner(num_clients=5, num_nodes=307, random_seed=seed)
        if distribution == "iid":
            parts = partitioner.partition_iid(self.data_mgr.train_data_raw, self.data_mgr.adj_matrix, self.data_mgr.laplacian)
        else:
            parts = partitioner.partition_non_iid(self.data_mgr.train_data_raw, self.data_mgr.adj_matrix, self.data_mgr.laplacian)

        train_norm = self.data_mgr.scaler.transform(self.data_mgr.train_data_raw)
        if self.quick:
            train_norm = train_norm[:1500]

        clients = []
        for p in parts:
            c_norm = train_norm[:, p.sensor_indices, :]
            x_c, y_c = generate_sliding_windows(c_norm, in_steps=12, out_steps=3)
            vc = VirtualClient(
                client_id=p.client_id,
                sensor_indices=p.sensor_indices,
                train_x=x_c,
                train_y=y_c,
                local_laplacian=p.local_laplacian,
                model_config=self.model_cfg,
                batch_size=self.batch_size,
                device=torch.device("cpu")
            )
            clients.append(vc)

        return clients

    def run_fedavg_trial(self, distribution: str, seed: int) -> Dict[str, Any]:
        """Runs Experiment C: Standard FedAvg Trial."""
        set_seed(seed)
        clients = self._setup_clients(distribution, seed)
        global_model = CompactSTGCN(**self.model_cfg)

        coordinator = FederatedCoordinator(
            global_model=global_model,
            clients=clients,
            data_manager=self.data_mgr,
            device=torch.device("cpu")
        )
        summary = coordinator.train_federated(
            num_rounds=self.rounds,
            local_epochs=1,
            local_lr=1e-3,
            experiment_name=f"fedavg_{distribution}_seed{seed}"
        )
        return summary

    def run_befa_trial(
        self,
        distribution: str,
        seed: int,
        budget_mb: float = 4.5,
        compress: bool = False,
        event_threshold: float = 0.25,
        max_update_interval: int = 2,
        exp_tag: str = "befa"
    ) -> Dict[str, Any]:
        """Runs Experiment D / Ablation: Proposed BEFA Framework Trial."""
        set_seed(seed)
        clients = self._setup_clients(distribution, seed)
        global_model = CompactSTGCN(**self.model_cfg)
        budget_bytes = int(budget_mb * 1024 * 1024)

        coordinator = BEFACoordinator(
            global_model=global_model,
            clients=clients,
            data_manager=self.data_mgr,
            total_budget_bytes=budget_bytes,
            event_threshold=event_threshold,
            max_update_interval=max_update_interval,
            beta_event=0.3,
            max_norm_clip=25.0,
            enable_compression=compress,
            device=torch.device("cpu")
        )

        comp_str = "int8" if compress else "uncompressed"
        exp_name = f"{exp_tag}_{distribution}_{comp_str}_b{budget_mb:.1f}_seed{seed}"

        summary = coordinator.train_befa(
            num_rounds=self.rounds,
            local_epochs=1,
            local_lr=1e-3,
            experiment_name=exp_name
        )
        return summary

    def log_result(
        self,
        paradigm: str,
        distribution: str,
        seed: int,
        budget_mb: float,
        compressed: bool,
        event_enabled: bool,
        total_bytes: int,
        test_metrics: Dict[str, float],
        notes: str = ""
    ):
        """Appends a standardized experiment record."""
        rec = {
            "paradigm": paradigm,
            "distribution": distribution,
            "seed": seed,
            "budget_mb": budget_mb,
            "compressed": compressed,
            "event_enabled": event_enabled,
            "total_bytes": total_bytes,
            "total_bytes_mb": total_bytes / (1024.0 ** 2),
            "mae_overall": test_metrics.get("mae_overall", 0.0),
            "rmse_overall": test_metrics.get("rmse_overall", 0.0),
            "mape_overall": test_metrics.get("mape_overall", 0.0),
            "mae_step_1_5m": test_metrics.get("mae_step_1_5m", 0.0),
            "rmse_step_1_5m": test_metrics.get("rmse_step_1_5m", 0.0),
            "mae_step_2_10m": test_metrics.get("mae_step_2_10m", 0.0),
            "rmse_step_2_10m": test_metrics.get("rmse_step_2_10m", 0.0),
            "mae_step_3_15m": test_metrics.get("mae_step_3_15m", 0.0),
            "rmse_step_3_15m": test_metrics.get("rmse_step_3_15m", 0.0),
            "notes": notes
        }
        self.all_records.append(rec)
        print(f"[Logged] {paradigm} ({distribution}, seed={seed}) -> MAE: {rec['mae_overall']:.2f}, RMSE: {rec['rmse_overall']:.2f}, MB: {rec['total_bytes_mb']:.2f}")

    def save_master_table(self):
        """Exports master results DataFrame to CSV."""
        df = pd.DataFrame(self.all_records)
        csv_path = os.path.join(self.results_dir, "master_experiment_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[Master Results Exported] Saved {len(df)} trials to: {csv_path}")
        return df


def main():
    parser = argparse.ArgumentParser(description="Run Full AIORI PS-003 Experimental Suite")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds (default: 42 123 999)")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per trial (default: 3)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--quick", action="store_true", help="Quick mode for fast execution")
    args = parser.parse_args()

    print("=" * 90)
    print(" AIORI PS-003: FULL EXPERIMENTAL SUITE & ABLATION STUDY")
    print(f" Seeds: {args.seeds} | Rounds: {args.rounds} | Batch Size: {args.batch_size} | Quick: {args.quick}")
    print("=" * 90)

    harness = ExperimentHarness(batch_size=args.batch_size, rounds=args.rounds, quick=args.quick)

    # 1. Experiment A: Persistence Baseline
    print("\n" + "=" * 60)
    print(" [1/5] RUNNING EXPERIMENT A: PERSISTENCE BASELINE")
    print("=" * 60)
    p_metrics = harness.run_persistence_baseline()
    for s in args.seeds:
        harness.log_result(
            paradigm="Persistence",
            distribution="N/A",
            seed=s,
            budget_mb=0.0,
            compressed=False,
            event_enabled=False,
            total_bytes=0,
            test_metrics=p_metrics,
            notes="Historical persistence baseline: last observed timestep copied forward"
        )

    # 2. Experiment B: Centralized STGCN
    print("\n" + "=" * 60)
    print(" [2/5] RUNNING EXPERIMENT B: CENTRALIZED STGCN")
    print("=" * 60)
    for s in args.seeds:
        cent_res = harness.run_centralized_trial(seed=s)
        harness.log_result(
            paradigm="Centralized STGCN",
            distribution="Global (307 sensors)",
            seed=s,
            budget_mb=0.0,
            compressed=False,
            event_enabled=False,
            total_bytes=0,
            test_metrics=cent_res["test_metrics"],
            notes=f"Upper bound benchmark (Wall time: {cent_res['total_train_time_sec']:.1f}s)"
        )

    # 3. Experiment C: Standard FedAvg (IID & Non-IID across seeds)
    print("\n" + "=" * 60)
    print(" [3/5] RUNNING EXPERIMENT C: STANDARD FEDAVG BASELINE")
    print("=" * 60)
    for dist in ["iid", "non_iid"]:
        for s in args.seeds:
            fed_res = harness.run_fedavg_trial(distribution=dist, seed=s)
            harness.log_result(
                paradigm="Standard FedAvg",
                distribution=dist.upper(),
                seed=s,
                budget_mb=0.0,
                compressed=False,
                event_enabled=False,
                total_bytes=fed_res["total_bytes"],
                test_metrics=fed_res["test_metrics"],
                notes="Standard sample-count weighted FedAvg baseline"
            )

    # 4. Experiment D: Proposed BEFA Framework (IID & Non-IID across seeds)
    print("\n" + "=" * 60)
    print(" [4/5] RUNNING EXPERIMENT D: PROPOSED BEFA FRAMEWORK")
    print("=" * 60)
    for dist in ["iid", "non_iid"]:
        for s in args.seeds:
            befa_res = harness.run_befa_trial(distribution=dist, seed=s, budget_mb=4.5)
            harness.log_result(
                paradigm="Proposed BEFA",
                distribution=dist.upper(),
                seed=s,
                budget_mb=4.5,
                compressed=False,
                event_enabled=True,
                total_bytes=befa_res["total_bytes"],
                test_metrics=befa_res["test_metrics"],
                notes=f"BEFA uncompressed (Budget 4.5MB, Deferred: {befa_res['deferred_transmissions']})"
            )

    # 5. Ablations: Compression, Budgets, and Event Triggering
    print("\n" + "=" * 60)
    print(" [5/5] RUNNING ABLATION STUDIES")
    print("=" * 60)
    # Ablation 1: Compression Enabled (int8)
    for s in args.seeds:
        comp_res = harness.run_befa_trial(distribution="non_iid", seed=s, budget_mb=3.0, compress=True, exp_tag="ablation_compression")
        harness.log_result(
            paradigm="BEFA + 8-bit Quantization",
            distribution="NON_IID",
            seed=s,
            budget_mb=3.0,
            compressed=True,
            event_enabled=True,
            total_bytes=comp_res["total_bytes"],
            test_metrics=comp_res["test_metrics"],
            notes="8-bit uniform quantization of model updates"
        )

    # Ablation 2: Budget Sensitivity (Tight 2.5 MB vs Standard 4.5 MB)
    tight_res = harness.run_befa_trial(distribution="non_iid", seed=42, budget_mb=2.5, exp_tag="ablation_tight_budget")
    harness.log_result(
        paradigm="BEFA (Tight Budget 2.5MB)",
        distribution="NON_IID",
        seed=42,
        budget_mb=2.5,
        compressed=False,
        event_enabled=True,
        total_bytes=tight_res["total_bytes"],
        test_metrics=tight_res["test_metrics"],
        notes=f"Tight budget cap: 2.5MB (Deferred: {tight_res['deferred_transmissions']})"
    )

    # Ablation 3: Event Scheduling Disabled (Blind Participation within Budget)
    blind_res = harness.run_befa_trial(distribution="non_iid", seed=42, budget_mb=4.5, event_threshold=0.0, exp_tag="ablation_no_events")
    harness.log_result(
        paradigm="BEFA (No Events / Blind)",
        distribution="NON_IID",
        seed=42,
        budget_mb=4.5,
        compressed=False,
        event_enabled=False,
        total_bytes=blind_res["total_bytes"],
        test_metrics=blind_res["test_metrics"],
        notes="Event triggering disabled (gamma=0.0)"
    )

    # Final Summary Table Export
    df_results = harness.save_master_table()
    print("\n" + "=" * 90)
    print(" EXPERIMENT SUITE COMPLETED SUCCESSFULLY")
    print("=" * 90)


if __name__ == "__main__":
    main()
