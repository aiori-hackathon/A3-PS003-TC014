"""
Centralized Training Pipeline for Compact STGCN on PEMS04.
Establishes the upper-bound baseline against which all federated experiments
(FedAvg and BEFA) are rigorously benchmarked.
"""

import os
import sys
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.stgcn import CompactSTGCN
from models.graph_layers import compute_chebyshev_polynomials
from data.dataset_loader import PEMS04DataManager
from evaluation.metrics import evaluate_forecast_metrics, evaluate_persistence_baseline


class CentralizedTrainer:
    """
    Centralized training and evaluation engine for STGCN on PEMS04.
    """
    def __init__(
        self,
        model: CompactSTGCN,
        data_manager: PEMS04DataManager,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "results/checkpoints",
        log_dir: str = "results/logs"
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data_manager = data_manager
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.SmoothL1Loss()  # Huber loss for robustness to traffic surges

        # Precompute and cache Chebyshev polynomials on device
        laplacian = self.data_manager.laplacian
        cheb_poly_np = compute_chebyshev_polynomials(laplacian, k_order=self.model.k_order)
        self.cheb_poly = torch.from_numpy(np.stack(cheb_poly_np, axis=0)).float().to(self.device)

        # Track history
        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "epoch_time_sec": []
        }
        self.best_val_loss = float("inf")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Runs one full training epoch across batches."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x, self.cheb_poly)
            loss = self.criterion(preds, y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluates model on validation or test DataLoader.
        Inverts predictions back to original physical scale (vehicles / 5-min)
        before computing MAE, RMSE, and MAPE.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(loader)

        all_preds = []
        all_targets = []

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            preds = self.model(x, self.cheb_poly)
            loss = self.criterion(preds, y)
            total_loss += loss.item()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / num_batches
        preds_norm = np.concatenate(all_preds, axis=0)
        targets_norm = np.concatenate(all_targets, axis=0)

        # Inverse transform to original scale (vehicles/5-min)
        preds_raw = self.data_manager.scaler.inverse_transform_flow(preds_norm)
        targets_raw = self.data_manager.scaler.inverse_transform_flow(targets_norm)

        # Compute physical metrics
        metrics = evaluate_forecast_metrics(preds_raw, targets_raw)

        return avg_loss, metrics, preds_raw, targets_raw

    def fit(
        self,
        epochs: int = 5,
        early_stopping_patience: int = 3,
        checkpoint_name: str = "centralized_best.pt"
    ) -> Dict[str, Any]:
        """Executes centralized model training with validation checkpointing."""
        train_loader, val_loader, test_loader = self.data_manager.get_dataloaders()
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        patience_counter = 0

        print(f"\n[Centralized Training] Starting {epochs} Epochs on {self.device}...")
        print(f"  Training Batches: {len(train_loader)} | Batch Size: {self.data_manager.batch_size}")
        print(f"  Validation Batches: {len(val_loader)} | Test Batches: {len(test_loader)}")
        print("-" * 80)
        print(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'Val MAE':<14} {'Val RMSE':<14} {'Time (s)':<10}")
        print("-" * 80)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics, _, _ = self.evaluate(val_loader)
            ep_time = time.time() - ep_start

            val_mae = val_metrics["mae_overall"]
            val_rmse = val_metrics["rmse_overall"]

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_mae)
            self.history["val_rmse"].append(val_rmse)
            self.history["epoch_time_sec"].append(ep_time)

            print(f"{epoch:<8} {train_loss:<14.4f} {val_loss:<14.4f} {val_mae:<14.2f} {val_rmse:<14.2f} {ep_time:<10.1f}")

            # Checkpoint best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics
                }, checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n[Early Stopping] Triggered at epoch {epoch} (Patience: {early_stopping_patience})")
                    break

        total_train_time = time.time() - start_time
        print("-" * 80)
        print(f"[Training Complete] Total Wall Time: {total_train_time:.1f} s ({total_train_time/60.0:.2f} min)")

        # Load best model for final test evaluation
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Best Checkpoint Loaded] From Epoch {ckpt['epoch']} (Val Loss: {ckpt['val_loss']:.4f})")

        # Evaluate on Test Set
        test_loss, test_metrics, _, targets_raw = self.evaluate(test_loader)
        
        # Compute Persistence Baseline on the exact test windows
        x_last_norm = self.data_manager.test_dataset.x[:, -1, :, 0].numpy()
        x_last_raw = self.data_manager.scaler.inverse_transform_flow(x_last_norm)
        persistence_preds = np.repeat(x_last_raw[:, np.newaxis, :], targets_raw.shape[1], axis=1)
        persistence_metrics = evaluate_forecast_metrics(persistence_preds, targets_raw)

        # Save training logs to CSV
        log_df = pd.DataFrame(self.history)
        log_csv = os.path.join(self.log_dir, "centralized_training_log.csv")
        log_df.to_csv(log_csv, index=False)

        summary = {
            "total_train_time_sec": total_train_time,
            "best_epoch": int(ckpt["epoch"]) if os.path.exists(checkpoint_path) else epochs,
            "best_val_loss": self.best_val_loss,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "persistence_metrics": persistence_metrics,
            "total_parameters": self.model.get_parameter_count()[0],
            "model_size_kb": self.model.get_model_size_kb(),
            "test_mae": test_metrics.get("mae_overall", 0.0),
            "test_rmse": test_metrics.get("rmse_overall", 0.0),
            "test_mape": test_metrics.get("mape_overall", 0.0),
            "persistence_test_rmse": persistence_metrics.get("rmse_overall", 0.0)
        }

        return summary


def train_centralized(
    data_path: str = "pems04.npz",
    csv_path: str = "data/adjacency/PEMS04.csv",
    num_epochs: int = 3,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42
) -> Dict[str, Any]:
    """Helper function to run centralized training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    data_mgr = PEMS04DataManager(
        npz_path=data_path,
        csv_path=csv_path,
        in_steps=12,
        out_steps=3,
        batch_size=batch_size
    ).prepare()
    
    model = CompactSTGCN(
        num_features=3,
        in_steps=12,
        out_steps=3,
        hidden_channels=32,
        k_order=2,
        kt=3,
        dropout=0.1
    )
    
    trainer = CentralizedTrainer(
        model=model,
        data_manager=data_mgr,
        learning_rate=lr
    )
    trainer.train = trainer.fit
    
    return trainer.fit(epochs=num_epochs)


