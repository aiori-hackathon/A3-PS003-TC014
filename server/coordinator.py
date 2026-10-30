"""
Federated Coordinator Module.
Orchestrates communication rounds, global parameter distribution,
local client training, sample-weighted aggregation, and byte accounting.
"""

import os
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
from server.fedavg import aggregate_fedavg
from data.dataset_loader import PEMS04DataManager
from evaluation.metrics import evaluate_forecast_metrics


class FederatedCoordinator:
    """
    Central server coordinating standard FedAvg rounds.
    """
    def __init__(
        self,
        global_model: CompactSTGCN,
        clients: List[VirtualClient],
        data_manager: PEMS04DataManager,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "results/checkpoints",
        log_dir: str = "results/logs"
    ):
        self.device = device or torch.device("cpu")
        self.global_model = global_model.to(self.device)
        self.clients = clients
        self.data_manager = data_manager
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Precompute global Chebyshev polynomials for server-side evaluation (N = 307)
        laplacian = self.data_manager.laplacian
        cheb_poly_np = compute_chebyshev_polynomials(laplacian, k_order=global_model.k_order)
        self.cheb_poly_global = torch.from_numpy(np.stack(cheb_poly_np, axis=0)).float().to(self.device)

        self.criterion = nn.SmoothL1Loss()

        # Cumulative communication tracking
        self.cumulative_bytes = 0

        # Detailed round history
        self.history: Dict[str, list] = {
            "round": [],
            "participating_clients": [],
            "download_bytes": [],
            "upload_bytes": [],
            "round_bytes": [],
            "cumulative_bytes": [],
            "client_train_loss_avg": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "round_time_sec": []
        }

    @torch.no_grad()
    def evaluate_global(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates current global model on validation or test DataLoader (307 sensors).
        Inverts predictions back to original physical scale (vehicles / 5-min).
        """
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

        # Invert to physical units (vehicles / 5-min)
        preds_raw = self.data_manager.scaler.inverse_transform_flow(preds_norm)
        targets_raw = self.data_manager.scaler.inverse_transform_flow(targets_norm)

        metrics = evaluate_forecast_metrics(preds_raw, targets_raw)
        return avg_loss, metrics

    def run_round(
        self,
        round_idx: int,
        local_epochs: int = 1,
        local_lr: float = 1e-3,
        participating_client_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Executes a single communication round of Federated Averaging:
        1. Distribute global model to participating clients (measure download bytes)
        2. Clients perform local training
        3. Collect client updates (measure upload bytes)
        4. Perform sample-count weighted FedAvg
        5. Update global model parameters
        6. Evaluate global model on validation set
        """
        start_time = time.time()
        global_state = {k: v.cpu().detach().clone() for k, v in self.global_model.state_dict().items()}

        # Select participating clients (default: all clients)
        active_clients = [c for c in self.clients if participating_client_ids is None or c.client_id in participating_client_ids]
        num_active = len(active_clients)

        download_bytes_round = 0
        upload_bytes_round = 0
        client_updates = []
        client_losses = []

        # Step 1 & 2: Broadcast global model and train locally
        for client in active_clients:
            down_bytes = client.receive_global_model(global_state)
            download_bytes_round += down_bytes

            c_loss = client.train_local(local_epochs=local_epochs, learning_rate=local_lr)
            client_losses.append(c_loss)

            # Step 3: Collect local updates
            c_dict, n_i, up_bytes = client.get_model_update()
            upload_bytes_round += up_bytes
            client_updates.append((c_dict, n_i))

        # Step 4: Sample-weighted FedAvg aggregation
        new_global_state = aggregate_fedavg(client_updates)

        # Step 5: Update global model
        self.global_model.load_state_dict(new_global_state)

        # Step 6: Evaluate global model on validation set
        _, val_loader, _ = self.data_manager.get_dataloaders()
        val_loss, val_metrics = self.evaluate_global(val_loader)

        round_time = time.time() - start_time
        round_bytes = download_bytes_round + upload_bytes_round
        self.cumulative_bytes += round_bytes

        avg_client_loss = float(np.mean(client_losses))
        val_mae = val_metrics["mae_overall"]
        val_rmse = val_metrics["rmse_overall"]

        # Record round metrics
        self.history["round"].append(round_idx)
        self.history["participating_clients"].append(num_active)
        self.history["download_bytes"].append(download_bytes_round)
        self.history["upload_bytes"].append(upload_bytes_round)
        self.history["round_bytes"].append(round_bytes)
        self.history["cumulative_bytes"].append(self.cumulative_bytes)
        self.history["client_train_loss_avg"].append(avg_client_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_mae"].append(val_mae)
        self.history["val_rmse"].append(val_rmse)
        self.history["round_time_sec"].append(round_time)

        round_summary = {
            "round": round_idx,
            "active_clients": num_active,
            "download_bytes": download_bytes_round,
            "upload_bytes": upload_bytes_round,
            "round_bytes": round_bytes,
            "cumulative_bytes": self.cumulative_bytes,
            "avg_train_loss": avg_client_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "round_time_sec": round_time
        }

        return round_summary

    def train_federated(
        self,
        num_rounds: int = 5,
        local_epochs: int = 1,
        local_lr: float = 1e-3,
        experiment_name: str = "fedavg"
    ) -> Dict[str, Any]:
        """Runs multiple communication rounds of FedAvg."""
        print(f"\n[Federated Learning: {experiment_name.upper()}] Starting {num_rounds} Rounds...")
        print(f"  Clients: {len(self.clients)} | Local Epochs: {local_epochs} | Local LR: {local_lr}")
        print("-" * 95)
        print(f"{'Round':<7} {'Clients':<9} {'Train Loss':<12} {'Val MAE':<12} {'Val RMSE':<12} {'Round KB':<12} {'Cumul MB':<12} {'Time (s)':<10}")
        print("-" * 95)

        for r in range(1, num_rounds + 1):
            res = self.run_round(
                round_idx=r,
                local_epochs=local_epochs,
                local_lr=local_lr
            )
            print(f"{r:<7} {res['active_clients']:<9} {res['avg_train_loss']:<12.4f} "
                  f"{res['val_mae']:<12.2f} {res['val_rmse']:<12.2f} "
                  f"{res['round_bytes']/1024.0:<12.1f} {res['cumulative_bytes']/(1024.0**2):<12.2f} "
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

        final_summary = {
            "experiment": experiment_name,
            "num_rounds": num_rounds,
            "total_bytes": self.cumulative_bytes,
            "total_bytes_mb": self.cumulative_bytes / (1024.0 ** 2),
            "final_val_mae": self.history["val_mae"][-1],
            "final_val_rmse": self.history["val_rmse"][-1],
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "checkpoint_path": ckpt_path,
            "log_csv": log_csv
        }

        return final_summary
