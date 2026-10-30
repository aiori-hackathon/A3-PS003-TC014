"""
Executable Script for Phase 4: Standard FedAvg Baseline Experiments.
Evaluates Federated STGCN under IID and Non-IID distributions with exact byte tracking.
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
from server.coordinator import FederatedCoordinator


def set_seed(seed: int = 42):
    """Ensures deterministic reproducibility across Python, NumPy, and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Run Standard FedAvg Baseline on PEMS04")
    parser.add_argument("--distribution", type=str, choices=["iid", "non_iid"], default="iid",
                        help="Client data distribution regime: iid or non_iid (default: iid)")
    parser.add_argument("--rounds", type=int, default=5, help="Number of communication rounds (default: 5)")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per round (default: 1)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for client training (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Client local learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--hidden_channels", type=int, default=32, help="STGCN hidden channels (default: 32)")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid testing (subsamples training set)")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 85)
    print(f" AIORI PS-003: FEDERATED LEARNING BASELINE (FedAvg - {args.distribution.upper()})")
    print(f" Distribution: {args.distribution.upper()} | Rounds: {args.rounds} | Local Epochs: {args.local_epochs} | Seed: {args.seed}")
    print("=" * 85)

    # 1. Prepare Global Data Manager
    print("\n[Step 1/5] Loading PEMS04 Dataset and Global Graph...")
    data_mgr = PEMS04DataManager(
        npz_path="pems04.npz",
        csv_path="data/adjacency/PEMS04.csv",
        in_steps=12,
        out_steps=3,
        batch_size=args.batch_size
    ).prepare()

    # 2. Partition Sensors across 5 Virtual Clients
    print(f"\n[Step 2/5] Creating 5 Virtual Clients under {args.distribution.upper()} Partitioning...")
    partitioner = FederatedPartitioner(num_clients=5, num_nodes=307, random_seed=args.seed)
    
    if args.distribution == "iid":
        partitions = partitioner.partition_iid(data_mgr.train_data_raw, data_mgr.adj_matrix, data_mgr.laplacian)
    else:
        partitions = partitioner.partition_non_iid(data_mgr.train_data_raw, data_mgr.adj_matrix, data_mgr.laplacian)

    # Transform normalized training data
    train_norm = data_mgr.scaler.transform(data_mgr.train_data_raw)
    if args.quick:
        print("[Quick Mode] Subsampling training data to 1,500 timesteps...")
        train_norm = train_norm[:1500]

    # Model configuration
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
    print("\n[Step 3/5] Instantiating Virtual Clients and Local Datasets:")
    clients = []
    for p in partitions:
        # Slice local sensors for this client
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
        print(f"  Client {p.client_id}: {p.num_sensors} sensors, {len(x_c)} samples | Mean Flow: {p.mean_flow:.2f} veh/5-min")

    # 4. Instantiate Global Server Model & Coordinator
    print("\n[Step 4/5] Initializing Server Global STGCN Model and Coordinator...")
    global_model = CompactSTGCN(**model_cfg)
    total_params, _ = global_model.get_parameter_count()
    size_kb = global_model.get_model_size_kb()
    print(f"  Global STGCN: {total_params:,} parameters | Uncompressed Payload: {size_kb:.2f} KB")

    exp_name = f"fedavg_{args.distribution}_seed{args.seed}"
    coordinator = FederatedCoordinator(
        global_model=global_model,
        clients=clients,
        data_manager=data_mgr,
        device=torch.device("cpu"),
        checkpoint_dir="results/checkpoints",
        log_dir="results/logs"
    )

    # 5. Launch Federated Training
    print(f"\n[Step 5/5] Executing Federated Training ({args.rounds} Rounds)...")
    summary = coordinator.train_federated(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.lr,
        experiment_name=exp_name
    )

    # 6. Display Final Metrics and Communication Summary
    print("\n" + "=" * 85)
    print(f" FEDERATED TRAINING COMPLETE ({exp_name.upper()})")
    print("=" * 85)
    print(f" Total Communication Rounds: {summary['num_rounds']}")
    print(f" Total Transmitted Bytes:    {summary['total_bytes']:,} bytes ({summary['total_bytes_mb']:.2f} MB)")
    print(f" Final Validation MAE:       {summary['final_val_mae']:.2f} veh/5-min")
    print(f" Final Validation RMSE:      {summary['final_val_rmse']:.2f} veh/5-min")

    test_met = summary["test_metrics"]
    print("-" * 85)
    print(" Global Model Test Set Performance (307 Sensors):")
    print(f"   Overall MAE:              {test_met['mae_overall']:.2f} veh/5-min")
    print(f"   Overall RMSE:             {test_met['rmse_overall']:.2f} veh/5-min")
    print(f"   Overall MAPE:             {test_met['mape_overall']:.2f} %")
    print(f"   Step 1 (5m)  MAE / RMSE:  {test_met['mae_step_1_5m']:.2f} / {test_met['rmse_step_1_5m']:.2f}")
    print(f"   Step 2 (10m) MAE / RMSE:  {test_met['mae_step_2_10m']:.2f} / {test_met['rmse_step_2_10m']:.2f}")
    print(f"   Step 3 (15m) MAE / RMSE:  {test_met['mae_step_3_15m']:.2f} / {test_met['rmse_step_3_15m']:.2f}")
    print("=" * 85)

    # Export structured metrics CSV
    os.makedirs("results/metrics", exist_ok=True)
    metrics_csv = f"results/metrics/{exp_name}_test_summary.csv"
    records = [{
        "experiment": exp_name,
        "distribution": args.distribution,
        "seed": args.seed,
        "num_rounds": args.rounds,
        "total_bytes": summary["total_bytes"],
        "total_bytes_mb": summary["total_bytes_mb"],
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
