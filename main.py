"""
Main CLI Entrypoint for AIORI Hackathon 3 PS-003
Project: Secure, Budgeted, Event-Aware Federated Traffic Learning (BEFA)

Allows running every phase of the project through unified CLI flags:
  1. inspect      - Inspect PEMS04 dataset and adjacency graph topology
  2. centralized  - Train centralized STGCN upper-bound baseline
  3. fedavg       - Train standard Federated Averaging baseline
  4. befa         - Train proposed BEFA framework
  5. test         - Run master unit and integration test suite
  6. experiments  - Run full multi-seed benchmark harness across seeds 42, 123, 999
  7. plots        - Generate publication-grade figures from CSV results
  8. all          - Execute tests, befa benchmark, and regenerate plots
"""

import os
import sys
import argparse
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def print_banner():
    banner = r"""
================================================================================
          AIORI HACKATHON 3 - PS-003: FEDERATED TRAFFIC LEARNING
   BEFA: Budget-Aware Event-Triggered Federated Aggregation with STGCN
================================================================================
    """
    print(banner)


def cmd_inspect(args):
    print("\n[Phase 1] Inspecting PEMS04 Dataset & Graph Topology...")
    from data.dataset_loader import load_raw_pems04, build_adjacency_matrix
    
    data = load_raw_pems04(args.data_path)
    print(f"\n[Raw Dataset] Loaded array from '{args.data_path}': shape={data.shape}, dtype={data.dtype}")
    print(f"  Total Timestamps: {data.shape[0]} (59 days @ 5-min intervals)")
    print(f"  Sensor Nodes:     {data.shape[1]}")
    print(f"  Channel Features: {data.shape[2]} (0: Flow, 1: Occupancy, 2: Speed)")
    print(f"  Flow stats: min={data[..., 0].min():.1f}, max={data[..., 0].max():.1f}, mean={data[..., 0].mean():.1f}")
    
    adj, lap = build_adjacency_matrix("data/adjacency/PEMS04.csv", num_nodes=data.shape[1])
    print(f"\n[Graph Topology] Loaded Adjacency Matrix: shape={adj.shape}, non-zero edges={(adj > 0).sum() - 307}")
    print(f"[Graph Topology] Normalized Symmetric Laplacian: shape={lap.shape}, dtype={lap.dtype}")
    print("[Success] Dataset and adjacency graph validated.\n")


def cmd_test(args):
    print("\n[Verification] Executing Master Test Suite...")
    from tests.run_all_tests import run_master_test_suite
    success = run_master_test_suite()
    if not success:
        print("\n[ERROR] One or more tests failed!")
        sys.exit(1)
    print("\n[Success] All unit and integration tests passed.\n")


def cmd_centralized(args):
    print(f"\n[Phase 3] Running Centralized STGCN Training (Epochs={args.epochs}, BatchSize={args.batch_size}, Seed={args.seed})...")
    from training.centralized import train_centralized
    results = train_centralized(
        data_path=args.data_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=0.001,
        seed=args.seed
    )
    print("\n[Centralized Results Summary]")
    print(f"  Persistence Test RMSE: {results['persistence_test_rmse']:.2f} veh/5-min")
    print(f"  Centralized Test MAE:  {results['test_mae']:.2f} veh/5-min")
    print(f"  Centralized Test RMSE: {results['test_rmse']:.2f} veh/5-min")
    print(f"  Centralized Test MAPE: {results['test_mape']:.2f}%")


def cmd_fedavg(args):
    print(f"\n[Phase 4] Running Standard FedAvg Baseline (Dist={args.dist}, Rounds={args.rounds}, Seed={args.seed})...")
    from evaluation.experiments import ExperimentHarness
    harness = ExperimentHarness(batch_size=args.batch_size, rounds=args.rounds, quick=args.quick)
    summary = harness.run_fedavg_trial(distribution=args.dist, seed=args.seed)
    print("\n[Standard FedAvg Results Summary]")
    print(f"  Total Bytes Transmitted: {summary['total_bytes']:,} bytes ({summary['total_bytes_mb']:.2f} MB)")
    print(f"  Final Test MAE:          {summary['test_metrics']['mae_overall']:.2f} veh/5-min")
    print(f"  Final Test RMSE:         {summary['test_metrics']['rmse_overall']:.2f} veh/5-min")
    print(f"  Final Test MAPE:         {summary['test_metrics']['mape_overall']:.2f}%")


def cmd_befa(args):
    print(f"\n[Phase 5] Running Proposed BEFA Framework (Dist={args.dist}, Budget={args.budget_mb}MB, Quantized={args.compress}, Seed={args.seed})...")
    from evaluation.experiments import ExperimentHarness
    harness = ExperimentHarness(batch_size=args.batch_size, rounds=args.rounds, quick=args.quick)
    summary = harness.run_befa_trial(
        distribution=args.dist,
        seed=args.seed,
        budget_mb=args.budget_mb,
        compress=args.compress,
        event_threshold=0.25,
        max_update_interval=2
    )
    print("\n[Proposed BEFA Results Summary]")
    print(f"  Total Transmitted:   {summary['total_bytes']:,} bytes ({summary['total_bytes_mb']:.2f} MB)")
    print(f"  Budget Utilization:  {(summary['total_bytes_mb'] / args.budget_mb) * 100:.1f}%")
    print(f"  Uploads Deferred:    {summary['deferred_transmissions']}")
    print(f"  Final Test MAE:      {summary['test_metrics']['mae_overall']:.2f} veh/5-min")
    print(f"  Final Test RMSE:     {summary['test_metrics']['rmse_overall']:.2f} veh/5-min")
    print(f"  Final Test MAPE:     {summary['test_metrics']['mape_overall']:.2f}%")


def cmd_experiments(args):
    print("\n[Phase 6] Running Automated Multi-Seed Experiment Harness...")
    from evaluation.experiments import run_all_experiments
    run_all_experiments(
        data_path=args.data_path,
        seeds=[42, 123, 999],
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        batch_size=args.batch_size
    )


def cmd_plots(args):
    print("\n[Phase 6] Generating Publication Figures from Master CSV...")
    from evaluation.plotting import plot_multi_seed_bars, plot_pareto_frontier
    plot_multi_seed_bars()
    plot_pareto_frontier()
    print("[Success] All figures updated in results/plots/\n")


def cmd_demo(args):
    print("\n[Demonstration] Executing Full Research Demonstration...")
    from demo import run_full_demo
    run_full_demo(quick=args.quick)


def cmd_all(args):
    print("\n[Executing Full End-to-End Pipeline]")
    cmd_test(args)
    cmd_befa(args)
    cmd_plots(args)
    print("\n[Pipeline Complete] Ready for evaluation and submission!\n")


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Unified CLI for BEFA (AIORI Hackathon 3 PS-003)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="befa",
        choices=["inspect", "test", "centralized", "fedavg", "befa", "experiments", "plots", "demo", "all"],
        help="Pipeline phase or action to execute"
    )
    parser.add_argument("--data_path", type=str, default="pems04.npz", help="Path to pems04.npz dataset")
    parser.add_argument("--dist", type=str, default="NON_IID", choices=["IID", "NON_IID"], help="Client partition distribution")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated global rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per client per round (or centralized epochs)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--budget_mb", type=float, default=4.5, help="Total communication budget cap in Megabytes")
    parser.add_argument("--compress", action="store_true", help="Enable 8-bit uniform quantization of model updates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--quick", action="store_true", help="Fast execution mode for demo/testing")

    args = parser.parse_args()

    mode_map = {
        "inspect": cmd_inspect,
        "test": cmd_test,
        "centralized": cmd_centralized,
        "fedavg": cmd_fedavg,
        "befa": cmd_befa,
        "experiments": cmd_experiments,
        "plots": cmd_plots,
        "demo": cmd_demo,
        "all": cmd_all
    }

    handler = mode_map.get(args.mode)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
