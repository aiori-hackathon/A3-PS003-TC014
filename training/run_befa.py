"""
Executable Script for Proposed BEFA Framework Experiments.
Executes Budget-Aware Event-Triggered Federated Learning with intelligent aggregation.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.stgcn import CompactSTGCN
from data.dataset_loader import PEMS04DataManager, generate_sliding_windows
from clients.partitioner import FederatedPartitioner
from clients.client import VirtualClient
from server.befa_coordinator import BEFACoordinator


def set_seed(seed: int = 42):
    """Ensures deterministic reproducibility across Python, NumPy, and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Run Proposed BEFA Framework on PEMS04")
    parser.add_argument("--distribution", type=str, choices=["iid", "non_iid"], default="non_iid",
                        help="Data distribution regime: iid or non_iid (default: non_iid)")
    parser.add_argument("--budget_mb", type=float, default=4.0, help="Hard communication budget in MB (default: 4.0)")
    parser.add_argument("--event_threshold", type=float, default=0.25, help="Event score threshold gamma (default: 0.25)")
    parser.add_argument("--max_update_interval", type=int, default=2, help="Max rounds before forced update tau_max (default: 2)")
    parser.add_argument("--beta_event", type=float, default=0.3, help="Event weighting scaling beta (default: 0.3)")
    parser.add_argument("--compress", action="store_true", help="Enable 8-bit quantized update compression")
    parser.add_argument("--rounds", type=int, default=5, help="Number of communication rounds (default: 5)")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per round (default: 1)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for client training (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Client local learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--hidden_channels", type=int, default=32, help="STGCN hidden channels (default: 32)")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid testing (subsamples training set)")
    args = parser.parse_args()

    set_seed(args.seed)

    comp_str = "int8" if args.compress else "uncompressed"
    exp_name = f"befa_{args.distribution}_{comp_str}_seed{args.seed}"
    budget_bytes = int(args.budget_mb * 1024 * 1024)

    print("=" * 95)
    print(f" AIORI PS-003: PROPOSED FRAMEWORK (BEFA - {args.distribution.upper()})")
    print(f" Budget: {args.budget_mb:.2f} MB ({budget_bytes:,} bytes) | Event Thresh: {args.event_threshold} | Max Interval: {args.max_update_interval}")
    print(f" Compression: {'ENABLED' if args.compress else 'DISABLED'} | Rounds: {args.rounds} | Seed: {args.seed}")
    print("=" * 95)

    # 1. Prepare Dataset
    print("\n[Step 1/5] Loading PEMS04 Dataset...")
    data_mgr = PEMS04DataManager(
        npz_path="pems04.npz",
        csv_path="data/adjacency/PEMS04.csv",
        in_steps=12,
        out_steps=3,
        batch_size=args.batch_size
    ).prepare()

    # 2. Partition Sensors across 5 Virtual Clients
    print(f"\n[Step 2/5] Partitioning Sensors ({args.distribution.upper()})...")
    partitioner = FederatedPartitioner(num_clients=5, num_nodes=307, random_seed=args.seed)
    if args.distribution == "iid":
        partitions = partitioner.partition_iid(data_mgr.train_data_raw, data_mgr.adj_matrix, data_mgr.laplacian)
    else:
        partitions = partitioner.partition_non_iid(data_mgr.train_data_raw, data_mgr.adj_matrix, data_mgr.laplacian)

    train_norm = data_mgr.scaler.transform(data_mgr.train_data_raw)
    if args.quick:
        print("[Quick Mode] Subsampling training data to 1,500 timesteps...")
        train_norm = train_norm[:1500]

    model_cfg = {
        "num_features": 3,
        "in_steps": 12,
        "out_steps": 3,
        "hidden_channels": args.hidden_channels,
        "k_order": 2,
        "kt": 3,
        "dropout": 0.1
    }

    # 3. Instantiate Virtual Clients
    print("\n[Step 3/5] Instantiating Virtual Clients...")
    clients = []
    for p in partitions:
        c_train_norm = train_norm[:, p.sensor_indices, :]
        x_c, y_c = generate_sliding_windows(c_train_norm, in_steps=12, out_steps=3)

        vc = VirtualClient(
            client_id=p.client_id,
            sensor_indices=p.sensor_indices,
            train_x=x_c,
            train_y=y_c,
            local_laplacian=p.local_laplacian,
            model_config=model_cfg,
            batch_size=args.batch_size,
            device=torch.device("cpu")
        )
        clients.append(vc)
        print(f"  Client {p.client_id}: {p.num_sensors} sensors, {len(x_c)} samples | Mean Flow: {p.mean_flow:.2f}")

    # 4. Instantiate Server Global STGCN Model and BEFA Coordinator
    print("\n[Step 4/5] Initializing Server Global STGCN Model and BEFA Coordinator...")
    global_model = CompactSTGCN(**model_cfg)
    total_params, _ = global_model.get_parameter_count()
    size_kb = global_model.get_model_size_kb()
    print(f"  Global STGCN: {total_params:,} parameters | Uncompressed Payload: {size_kb:.2f} KB")

    coordinator = BEFACoordinator(
        global_model=global_model,
        clients=clients,
        data_manager=data_mgr,
        total_budget_bytes=budget_bytes,
        event_threshold=args.event_threshold,
        max_update_interval=args.max_update_interval,
        beta_event=args.beta_event,
        max_norm_clip=5.0,
        enable_compression=args.compress,
        device=torch.device("cpu"),
        checkpoint_dir="results/checkpoints",
        log_dir="results/logs"
    )

    # 5. Launch BEFA Training
    print(f"\n[Step 5/5] Executing BEFA Training ({args.rounds} Rounds)...")
    summary = coordinator.train_befa(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.lr,
        experiment_name=exp_name
    )

    # 6. Display Final Metrics and Budget Report
    print("\n" + "=" * 95)
    print(f" BEFA TRAINING COMPLETE ({exp_name.upper()})")
    print("=" * 95)
    print(f" Total Completed Rounds:     {summary['num_rounds']}")
    print(f" Total Transmitted Bytes:    {summary['total_bytes']:,} bytes ({summary['total_bytes_mb']:.2f} MB)")
    print(f" Deferred Transmissions:     {summary['deferred_transmissions']}")
    print(f" Hard Budget Exhausted:      {summary['budget_exhausted']}")
    print(f" Final Validation MAE:       {summary['final_val_mae']:.2f} veh/5-min")
    print(f" Final Validation RMSE:      {summary['final_val_rmse']:.2f} veh/5-min")

    test_met = summary["test_metrics"]
    print("-" * 95)
    print(" Global Model Test Set Performance (307 Sensors):")
    print(f"   Overall MAE:              {test_met['mae_overall']:.2f} veh/5-min")
    print(f"   Overall RMSE:             {test_met['rmse_overall']:.2f} veh/5-min")
    print(f"   Overall MAPE:             {test_met['mape_overall']:.2f} %")
    print(f"   Step 1 (5m)  MAE / RMSE:  {test_met['mae_step_1_5m']:.2f} / {test_met['rmse_step_1_5m']:.2f}")
    print(f"   Step 2 (10m) MAE / RMSE:  {test_met['mae_step_2_10m']:.2f} / {test_met['rmse_step_2_10m']:.2f}")
    print(f"   Step 3 (15m) MAE / RMSE:  {test_met['mae_step_3_15m']:.2f} / {test_met['rmse_step_3_15m']:.2f}")
    print("=" * 95)

    # Export structured metrics CSV
    os.makedirs("results/metrics", exist_ok=True)
    metrics_csv = f"results/metrics/{exp_name}_test_summary.csv"
    records = [{
        "experiment": exp_name,
        "distribution": args.distribution,
        "compressed": args.compress,
        "seed": args.seed,
        "budget_mb": args.budget_mb,
        "num_rounds": summary["num_rounds"],
        "total_bytes": summary["total_bytes"],
        "total_bytes_mb": summary["total_bytes_mb"],
        "deferred_transmissions": summary["deferred_transmissions"],
        "budget_exhausted": summary["budget_exhausted"],
        "val_mae": summary["final_val_mae"],
        "val_rmse": summary["final_val_rmse"],
        "test_mae_overall": test_met["mae_overall"],
        "test_rmse_overall": test_met["rmse_overall"],
        "test_mape_overall": test_met["mape_overall"],
        "test_mae_5m": test_met["mae_step_1_5m"],
        "test_rmse_5m": test_met["rmse_step_1_5m"],
        "test_mae_10m": test_met["mae_step_2_10m"],
        "test_rmse_10m": test_met["rmse_step_2_10m"],
        "test_mae_15m": test_met["mae_step_3_15m"],
        "test_rmse_15m": test_met["rmse_step_3_15m"],
    }]
    pd.DataFrame(records).to_csv(metrics_csv, index=False)
    print(f"[Exported] Metrics saved to: {metrics_csv}")
    print(f"[Exported] Round log saved to: {summary['log_csv']}")
    print(f"[Exported] Checkpoint saved to: {summary['checkpoint_path']}")


if __name__ == "__main__":
    main()
