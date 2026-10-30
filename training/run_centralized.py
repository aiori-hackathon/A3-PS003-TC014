"""
Executable script to run Phase 3: Centralized Baseline Training & Evaluation.
Compares CompactSTGCN against the Persistence Baseline on PEMS04.
Saves metrics to results/metrics/centralized_metrics.csv and checkpoint to results/checkpoints/.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.stgcn import CompactSTGCN
from data.dataset_loader import PEMS04DataManager
from training.centralized import CentralizedTrainer


def set_seed(seed: int = 42):
    """Ensures deterministic reproducibility across Python, NumPy, and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train Centralized CompactSTGCN Baseline on PEMS04")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--hidden_channels", type=int, default=32, help="Hidden channels (default: 32)")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid prototyping (subsamples train data)")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print(" AIORI PS-003: CENTRALIZED BASELINE TRAINING (COMPACT STGCN)")
    print(f" Seed: {args.seed} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 80)

    # 1. Prepare Dataset
    print("\n[Step 1/4] Preparing PEMS04 Dataset...")
    data_mgr = PEMS04DataManager(
        npz_path="pems04.npz",
        csv_path="data/adjacency/PEMS04.csv",
        in_steps=12,
        out_steps=3,
        batch_size=args.batch_size
    ).prepare()

    # Optional quick prototyping subsample
    if args.quick:
        print("[Quick Mode] Subsampling training windows for rapid prototyping (first 1,500 samples)...")
        data_mgr.train_dataset.x = data_mgr.train_dataset.x[:1500]
        data_mgr.train_dataset.y = data_mgr.train_dataset.y[:1500]
        data_mgr.val_dataset.x = data_mgr.val_dataset.x[:500]
        data_mgr.val_dataset.y = data_mgr.val_dataset.y[:500]
        data_mgr.test_dataset.x = data_mgr.test_dataset.x[:1000]
        data_mgr.test_dataset.y = data_mgr.test_dataset.y[:1000]

    # 2. Instantiate Model
    print("\n[Step 2/4] Initializing CompactSTGCN Model...")
    model = CompactSTGCN(
        num_features=3,
        in_steps=12,
        out_steps=3,
        hidden_channels=args.hidden_channels,
        k_order=2,
        kt=3,
        dropout=0.1
    )
    total_params, _ = model.get_parameter_count()
    size_kb = model.get_model_size_kb()
    print(f"  Total Parameters: {total_params:,} | Uncompressed Size: {size_kb:.2f} KB")

    # 3. Train Centralized Baseline
    print("\n[Step 3/4] Launching Centralized Training...")
    trainer = CentralizedTrainer(
        model=model,
        data_manager=data_mgr,
        learning_rate=args.lr,
        weight_decay=1e-4,
        checkpoint_dir="results/checkpoints",
        log_dir="results/logs"
    )

    summary = trainer.fit(epochs=args.epochs, early_stopping_patience=3)

    # 4. Display and Save Final Baseline Metrics
    print("\n[Step 4/4] Comparing Centralized STGCN vs. Persistence Baseline on Test Set:")
    print("=" * 80)
    print(f"{'Horizon / Step':<24} {'Metric':<10} {'Persistence':<18} {'Centralized STGCN':<18} {'Gain':<12}")
    print("-" * 80)

    p_met = summary["persistence_metrics"]
    s_met = summary["test_metrics"]

    horizons = [
        ("Step 1 (5 mins)", "step_1_5m"),
        ("Step 2 (10 mins)", "step_2_10m"),
        ("Step 3 (15 mins)", "step_3_15m"),
        ("Overall (15 mins)", "overall")
    ]

    records = []
    for h_label, h_key in horizons:
        p_mae = p_met[f"mae_{h_key}"]
        s_mae = s_met[f"mae_{h_key}"]
        mae_gain = ((p_mae - s_mae) / p_mae) * 100.0

        p_rmse = p_met[f"rmse_{h_key}"]
        s_rmse = s_met[f"rmse_{h_key}"]
        rmse_gain = ((p_rmse - s_rmse) / p_rmse) * 100.0

        p_mape = p_met[f"mape_{h_key}"]
        s_mape = s_met[f"mape_{h_key}"]

        print(f"{h_label:<24} {'MAE':<10} {p_mae:<18.2f} {s_mae:<18.2f} {mae_gain:+6.2f}%")
        print(f"{h_label:<24} {'RMSE':<10} {p_rmse:<18.2f} {s_rmse:<18.2f} {rmse_gain:+6.2f}%")
        print(f"{h_label:<24} {'MAPE':<10} {p_mape:<17.2f}% {s_mape:<17.2f}%")
        print("-" * 80)

        records.append({
            "horizon": h_label,
            "persistence_mae": p_mae,
            "centralized_mae": s_mae,
            "mae_gain_percent": mae_gain,
            "persistence_rmse": p_rmse,
            "centralized_rmse": s_rmse,
            "rmse_gain_percent": rmse_gain,
            "persistence_mape": p_mape,
            "centralized_mape": s_mape
        })

    # Save to CSV
    os.makedirs("results/metrics", exist_ok=True)
    metrics_csv = "results/metrics/centralized_vs_persistence.csv"
    pd.DataFrame(records).to_csv(metrics_csv, index=False)
    print(f"\n[Saved Metrics] Exported to: {metrics_csv}")
    print(f"[Saved Log] Training history: results/logs/centralized_training_log.csv")
    print(f"[Saved Checkpoint] Model state: results/checkpoints/centralized_best.pt")
    print("=" * 80)


if __name__ == "__main__":
    main()
