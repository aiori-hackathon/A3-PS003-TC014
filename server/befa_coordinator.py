"""
Integrated BEFA (Budget-Aware Event-Triggered Federated Aggregation) Coordinator.
Combines:
  1. Event-aware client participation triggering
  2. Hard communication budget enforcement & deferral
  3. L2 norm clipping and security validation
  4. Intelligent validation-informed BEFA aggregation
  5. Optional 8-bit quantized payload compression
  6. Resource cleanup and garbage collection
"""

import os
import gc
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.stgcn import CompactSTGCN
from models.graph_layers import compute_chebyshev_polynomials
from clients.client import VirtualClient
from communication.budget_manager import CommunicationBudgetManager
from communication.event_scheduler import EventScheduler
from communication.serialization import measure_payload_bytes
from communication.compression import compress_state_dict, decompress_state_dict
from security.update_validator import UpdateValidator
from server.proposed_aggregation import aggregate_befa
from data.dataset_loader import PEMS04DataManager
from evaluation.metrics import evaluate_forecast_metrics


class BEFACoordinator:
    """
    Central server coordinator for the proposed BEFA framework.
    """
    def __init__(
        self,
        global_model: CompactSTGCN,
        clients: List[VirtualClient],
        data_manager: PEMS04DataManager,
        total_budget_bytes: int = 4_000_000,   # Default ~4 MB hard budget
        event_threshold: float = 0.25,
        max_update_interval: int = 2,
        beta_event: float = 0.3,
        max_norm_clip: float = 25.0,
        enable_compression: bool = False,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "results/checkpoints",
        log_dir: str = "results/logs"
    ):
        self.device = device or torch.device("cpu")
        self.global_model = global_model.to(self.device)
        self.clients = clients
        self.data_manager = data_manager
        self.beta_event = beta_event
        self.enable_compression = enable_compression
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # 1. Budget Manager
        self.budget_mgr = CommunicationBudgetManager(total_budget_bytes=total_budget_bytes)

        # 2. Event Scheduler
        self.scheduler = EventScheduler(
            num_clients=len(clients),
            event_threshold=event_threshold,
            max_update_interval=max_update_interval
        )
        # Register initial client baselines
        for c in clients:
            self.scheduler.register_client_baseline(c.client_id, initial_mean_flow=200.0, initial_val_loss=1.0)

        # 3. Security & Update Validator
        self.validator = UpdateValidator(max_norm_clip=max_norm_clip, max_staleness_rounds=2)

        # 4. Global Graph Laplacian & Cheb Poly for evaluation
        laplacian = data_manager.laplacian
        cheb_poly_np = compute_chebyshev_polynomials(laplacian, k_order=global_model.k_order)
        self.cheb_poly_global = torch.from_numpy(np.stack(cheb_poly_np, axis=0)).float().to(self.device)

        self.criterion = nn.SmoothL1Loss()

        # Detailed round history
        self.history: Dict[str, list] = {
            "round": [],
            "scheduled_clients": [],
            "accepted_clients": [],
            "download_bytes": [],
            "upload_bytes": [],
            "round_bytes": [],
            "cumulative_bytes": [],
            "remaining_budget": [],
            "budget_exhausted": [],
            "client_train_loss_avg": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "round_time_sec": []
        }

    @torch.no_grad()
    def evaluate_global(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluates global model on validation/test set in physical units (vehicles / 5-min)."""
        self.global_model.eval()
        total_loss = 0.0
        num_batches = len(loader)

        all_preds = []
        all_targets = []

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            preds = self.global_model(x, self.cheb_poly_global)
            loss = self.criterion(preds, y)
            total_loss += loss.item()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / num_batches
        preds_norm = np.concatenate(all_preds, axis=0)
        targets_norm = np.concatenate(all_targets, axis=0)

        preds_raw = self.data_manager.scaler.inverse_transform_flow(preds_norm)
        targets_raw = self.data_manager.scaler.inverse_transform_flow(targets_norm)

        metrics = evaluate_forecast_metrics(preds_raw, targets_raw)
        return avg_loss, metrics

    def run_round(
        self,
        round_idx: int,
        local_epochs: int = 1,
        local_lr: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Executes a single communication round under the BEFA framework:
        1. Query Event Scheduler for client participation eligibility.
        2. Verify remaining budget capacity before dispatching models.
        3. Selected clients receive global model and train locally.
        4. Validate client updates (shapes, finiteness, L2 clipping, freshness).
        5. Aggregate accepted updates via BEFA weighting.
        6. Update global model and log communication accounting.
        7. Clean temporary memory resources.
        """
        start_time = time.time()
        global_state = {k: v.cpu().detach().clone() for k, v in self.global_model.state_dict().items()}

        raw_payload_bytes = measure_payload_bytes(global_state)
        # If compression is enabled, measure compressed payload
        if self.enable_compression:
            comp_bytes = len(compress_state_dict(global_state))
            down_unit_bytes = comp_bytes
            up_unit_bytes = comp_bytes
        else:
            down_unit_bytes = raw_payload_bytes
            up_unit_bytes = raw_payload_bytes

        # 1. Event-aware scheduling evaluation
        scheduled_clients = []
        client_event_scores = {}
        all_devs = [c.get_recent_traffic_deviation() for c in self.clients]
        mean_dev = float(np.mean(all_devs)) if all_devs else 0.05

        for client in self.clients:
            c_flow_dev = client.get_recent_traffic_deviation()
            # Relative volatility score (1.0 = average volatility across network)
            score = float(c_flow_dev / (mean_dev + 1e-5))
            client_event_scores[client.client_id] = score

            should_participate, reason = self.scheduler.evaluate_participation(
                client_id=client.client_id,
                current_round=round_idx,
                event_score=score
            )
            if should_participate:
                scheduled_clients.append(client)

        # 2. Budget Enforcement: check capacity for scheduled clients
        accepted_for_transmission = []
        for client in scheduled_clients:
            if self.budget_mgr.can_accommodate(down_unit_bytes + up_unit_bytes):
                # Allocate budget
                allocated = self.budget_mgr.request_transmission(down_unit_bytes, up_unit_bytes)
                if allocated:
                    accepted_for_transmission.append(client)
            else:
                self.budget_mgr.deferred_transmissions_count += 1

        # 3. Local Training & Update Collection
        accepted_updates = []
        client_losses = []
        download_bytes_round = 0
        upload_bytes_round = 0

        for client in accepted_for_transmission:
            # Broadcast global model
            client.receive_global_model(global_state)
            download_bytes_round += down_unit_bytes

            # Train locally
            c_loss = client.train_local(local_epochs=local_epochs, learning_rate=local_lr)
            client_losses.append(c_loss)

            # Export local update
            c_dict, n_i, up_bytes = client.get_model_update()
            
            # Optional compression simulation
            if self.enable_compression:
                c_payload = compress_state_dict(c_dict)
                c_dict = decompress_state_dict(c_payload)
                upload_bytes_round += len(c_payload)
            else:
                upload_bytes_round += up_bytes

            # 4. Security & Quality Validation (L2 norm clipping, finiteness, staleness)
            is_valid, bounded_dict, val_reason = self.validator.validate_and_clip(
                client_id=client.client_id,
                client_state=c_dict,
                global_state=global_state,
                client_round=round_idx,
                current_round=round_idx
            )

            if is_valid and bounded_dict is not None:
                metadata = {
                    "num_samples": n_i,
                    "event_score": client_event_scores[client.client_id],
                    "val_loss": c_loss
                }
                accepted_updates.append((bounded_dict, metadata))

        # 5. BEFA Aggregation (if any updates accepted)
        if accepted_updates:
            new_global_state, weights = aggregate_befa(accepted_updates, beta=self.beta_event)
            self.global_model.load_state_dict(new_global_state)

        # 6. Record participation in Scheduler
        participated_ids = [c.client_id for c in accepted_for_transmission]
        self.scheduler.record_round_completion(participated_ids, round_idx)

        # 7. Evaluate Global Model on Validation set
        _, val_loader, _ = self.data_manager.get_dataloaders()
        val_loss, val_metrics = self.evaluate_global(val_loader)

        round_time = time.time() - start_time
        round_bytes = download_bytes_round + upload_bytes_round
        avg_client_loss = float(np.mean(client_losses)) if client_losses else 0.0

        # Log round history
        self.history["round"].append(round_idx)
        self.history["scheduled_clients"].append(len(scheduled_clients))
        self.history["accepted_clients"].append(len(accepted_for_transmission))
        self.history["download_bytes"].append(download_bytes_round)
        self.history["upload_bytes"].append(upload_bytes_round)
        self.history["round_bytes"].append(round_bytes)
        self.history["cumulative_bytes"].append(self.budget_mgr.total_used_bytes)
        self.history["remaining_budget"].append(self.budget_mgr.get_remaining_budget())
        self.history["budget_exhausted"].append(self.budget_mgr.is_exhausted())
        self.history["client_train_loss_avg"].append(avg_client_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_mae"].append(val_metrics["mae_overall"])
        self.history["val_rmse"].append(val_metrics["rmse_overall"])
        self.history["round_time_sec"].append(round_time)

        # 8. Explicit Resource Cleanup
        del global_state
        if accepted_updates:
            del accepted_updates
        gc.collect()

        return {
            "round": round_idx,
            "scheduled": len(scheduled_clients),
            "participated": len(accepted_for_transmission),
            "round_bytes": round_bytes,
            "cumulative_bytes": self.budget_mgr.total_used_bytes,
            "remaining_budget": self.budget_mgr.get_remaining_budget(),
            "val_mae": val_metrics["mae_overall"],
            "val_rmse": val_metrics["rmse_overall"],
            "round_time_sec": round_time
        }

    def train_befa(
        self,
        num_rounds: int = 5,
        local_epochs: int = 1,
        local_lr: float = 1e-3,
        experiment_name: str = "befa"
    ) -> Dict[str, Any]:
        """Runs multi-round federated training under BEFA budget constraints."""
        print(f"\n[BEFA Framework: {experiment_name.upper()}] Starting {num_rounds} Rounds...")
        print(f"  Hard Budget: {self.budget_mgr.total_budget_bytes:,} bytes ({self.budget_mgr.total_budget_bytes/(1024**2):.2f} MB)")
        print(f"  Compression: {'ENABLED (8-bit)' if self.enable_compression else 'DISABLED'}")
        print("-" * 105)
        print(f"{'Round':<7} {'Sched':<7} {'Active':<8} {'Round KB':<12} {'Cumul MB':<12} {'Remain MB':<12} {'Val MAE':<12} {'Val RMSE':<12} {'Time (s)':<10}")
        print("-" * 105)

        for r in range(1, num_rounds + 1):
            if self.budget_mgr.is_exhausted():
                print(f"[Budget Stopping] Hard budget exhausted before round {r}. Terminating training.")
                break

            res = self.run_round(round_idx=r, local_epochs=local_epochs, local_lr=local_lr)
            print(f"{r:<7} {res['scheduled']:<7} {res['participated']:<8} "
                  f"{res['round_bytes']/1024.0:<12.1f} {res['cumulative_bytes']/(1024.0**2):<12.2f} "
                  f"{res['remaining_budget']/(1024.0**2):<12.2f} "
                  f"{res['val_mae']:<12.2f} {res['val_rmse']:<12.2f} "
                  f"{res['round_time_sec']:<10.1f}")

        # Save Checkpoint
        ckpt_path = os.path.join(self.checkpoint_dir, f"{experiment_name}_final.pt")
        torch.save(self.global_model.state_dict(), ckpt_path)

        # Final Test Evaluation
        _, _, test_loader = self.data_manager.get_dataloaders()
        test_loss, test_metrics = self.evaluate_global(test_loader)

        # Save History to CSV
        log_csv = os.path.join(self.log_dir, f"{experiment_name}_rounds_log.csv")
        pd.DataFrame(self.history).to_csv(log_csv, index=False)

        summary = {
            "experiment": experiment_name,
            "num_rounds": len(self.history["round"]),
            "total_bytes": self.budget_mgr.total_used_bytes,
            "total_bytes_mb": self.budget_mgr.total_used_bytes / (1024.0 ** 2),
            "deferred_transmissions": self.budget_mgr.deferred_transmissions_count,
            "budget_exhausted": self.budget_mgr.is_exhausted(),
            "final_val_mae": self.history["val_mae"][-1] if self.history["val_mae"] else 0.0,
            "final_val_rmse": self.history["val_rmse"][-1] if self.history["val_rmse"] else 0.0,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "checkpoint_path": ckpt_path,
            "log_csv": log_csv
        }

        return summary


# Alias for backwards compatibility
BefaCoordinator = BEFACoordinator

