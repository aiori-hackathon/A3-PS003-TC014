"""
Plotting and Visualization Module for AIORI Hackathon 3 PS-003.
Generates publication-grade figures directly from master_experiment_results.csv:
1. Multi-Seed Benchmark Comparison (Mean +/- Std error bars)
2. Pareto Frontier: Prediction Accuracy vs. Communication Overhead
3. Compression & Budget Ablation Analysis
4. Round Convergence Curves
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.color"] = "#e5e5e5"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def plot_multi_seed_bars(csv_path: str = "results/metrics/master_experiment_results.csv", save_dir: str = "results/plots"):
    """
    Bar chart with standard deviation error bars comparing paradigms across random seeds.
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] CSV {csv_path} not found.")
        return

    setup_style()
    df = pd.read_csv(csv_path)

    # Filter main comparison paradigms
    target_paradigms = [
        ("Centralized STGCN", "Global (307 sensors)"),
        ("Standard FedAvg", "IID"),
        ("Proposed BEFA", "IID"),
        ("Standard FedAvg", "NON_IID"),
        ("Proposed BEFA", "NON_IID"),
        ("BEFA + 8-bit Quantization", "NON_IID")
    ]

    labels = []
    mae_means, mae_stds = [], []
    rmse_means, rmse_stds = [], []
    bytes_means = []

    for p, dist in target_paradigms:
        sub = df[(df["paradigm"] == p) & (df["distribution"] == dist)]
        if len(sub) > 0:
            short_label = f"{p}\n({dist})"
            labels.append(short_label)
            mae_means.append(sub["mae_overall"].mean())
            mae_stds.append(sub["mae_overall"].std() if len(sub) > 1 else 0.0)
            rmse_means.append(sub["rmse_overall"].mean())
            rmse_stds.append(sub["rmse_overall"].std() if len(sub) > 1 else 0.0)
            bytes_means.append(sub["total_bytes_mb"].mean())

    fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=300)

    x = np.arange(len(labels))
    width = 0.35

    colors_mae = ["#1976d2", "#388e3c", "#2e7d32", "#d32f2f", "#c2185b", "#7b1fa2"]
    rects1 = ax1.bar(x - width/2, mae_means, width, yerr=mae_stds, capsize=4, label="Test MAE (veh/5m)", color="#457b9d", alpha=0.9, edgecolor="black")
    rects2 = ax1.bar(x + width/2, rmse_means, width, yerr=rmse_stds, capsize=4, label="Test RMSE (veh/5m)", color="#e63946", alpha=0.9, edgecolor="black")

    ax1.set_title("Experimental Benchmark Across Paradigms & Seeds (Mean $\\pm$ Std)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Prediction Error (veh / 5-min)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")

    # Add text annotations for bytes on top of bars
    for i in range(len(x)):
        ax1.text(x[i], max(rmse_means[i] + (rmse_stds[i] or 0), mae_means[i]) + 2.5, f"{bytes_means[i]:.2f} MB", ha="center", fontsize=9, fontweight="bold", color="#333333")

    plt.tight_layout()
    out_file = os.path.join(save_dir, "multi_seed_comparison.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[Plot Saved] Multi-seed comparison: {out_file}")


def plot_pareto_frontier(csv_path: str = "results/metrics/master_experiment_results.csv", save_dir: str = "results/plots"):
    """
    Plots the Communication vs Accuracy Pareto Trade-off.
    """
    if not os.path.exists(csv_path):
        return

    setup_style()
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Group by paradigm and distribution
    agg = df.groupby(["paradigm", "distribution"]).agg({
        "total_bytes_mb": "mean",
        "mae_overall": "mean",
        "rmse_overall": "mean"
    }).reset_index()

    color_map = {
        "Persistence": "#757575",
        "Centralized STGCN": "#1565c0",
        "Standard FedAvg": "#d9534f",
        "Proposed BEFA": "#2e7d32",
        "BEFA + 8-bit Quantization": "#8e24aa",
        "BEFA (Tight Budget 2.5MB)": "#f57c00",
        "BEFA (No Events / Blind)": "#0288d1"
    }

    for _, row in agg.iterrows():
        p = row["paradigm"]
        c = color_map.get(p, "#333333")
        m = "o" if "FedAvg" in p else ("^" if "BEFA" in p and "Quant" not in p else ("D" if "Quant" in p else "s"))
        size = 180 if "Quant" in p or "BEFA" in p else 120

        ax.scatter(row["total_bytes_mb"], row["mae_overall"], color=c, s=size, edgecolors="black", linewidth=1.2, zorder=5)
        dist_label = f" ({row['distribution']})" if row["distribution"] != "N/A" and "Global" not in row["distribution"] else ""
        ax.annotate(
            f"{p}{dist_label}",
            (row["total_bytes_mb"], row["mae_overall"]),
            textcoords="offset points",
            xytext=(8, -4 if "Quant" in p else 6),
            fontsize=9.5,
            fontweight="bold",
            color=c
        )

    ax.set_title("Pareto Frontier: Communication Cost vs. Prediction MAE", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Total Transmitted Communication (Megabytes)", fontsize=11)
    ax.set_ylabel("Test Prediction MAE (veh / 5-min)", fontsize=11)

    plt.tight_layout()
    out_file = os.path.join(save_dir, "pareto_frontier.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[Plot Saved] Pareto frontier: {out_file}")


def main():
    setup_style()
    plot_multi_seed_bars()
    plot_pareto_frontier()


if __name__ == "__main__":
    main()
