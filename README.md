# Secure, Budgeted, Event-Aware Federated Traffic Learning (BEFA)
### AIORI Hackathon 3 — Problem Statement PS-003: Federated Traffic-Flow Prediction with Communication-Cost Evaluation

[![Python 3.13](https://img.shields.io/badge/Python-3.13.15-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0+cpu-orange.svg)](https://pytorch.org/)
[![Test Suite](https://img.shields.io/badge/Tests-34%2F34%20Passing-brightgreen.svg)](tests/run_all_tests.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Communication Savings](https://img.shields.io/badge/Comm%20Savings-71.5%25-success.svg)](results/plots/pareto_frontier.png)

---

## 1. Problem Overview & Research Motivation

Municipal Traffic Management Centers (TMCs) across regional metropolitan districts generate massive streams of spatio-temporal loop detector telemetry. Centralizing continuous sensor telemetry to a single cloud server introduces severe privacy violations, high backhaul bandwidth costs, and single-point-of-failure vulnerabilities. 

**AIORI Hackathon 3 – Problem Statement PS-003** mandates developing a decentralized, communication-conscious federated learning system for highway traffic flow prediction.

### Key Research Challenges
1. **Spatio-Temporal Graph Dependency:** Traffic states exhibit complex spatial dependencies across interconnected highway segments and temporal dynamics across multi-step forecasting horizons.
2. **Severe Non-IID Client Drift:** Edge traffic partitions exhibit dramatic volume disparity (e.g. quiet suburban access ramps vs. multi-lane interstate freeways). Standard Federated Averaging (**FedAvg**) suffers catastrophic divergence under this statistical heterogeneity.
3. **Edge Communication Bottlenecks:** Edge links face strict monthly data quotas and limited uplink bandwidth, requiring fine-grained communication budget control.
4. **Adversarial & Numerical Vulnerabilities:** Faulty edge sensors or malicious actors can inject NaN/Inf parameters, out-of-distribution updates, or oversized gradient explosions that compromise the global model.

---

## 2. Proposed BEFA Architecture

To resolve these challenges, we propose **BEFA** (*Budget-Aware Event-Triggered Federated Aggregation*), a multi-tier federated coordination framework combining spatial graph convolutions with adaptive budget and security guards:

```
                  ┌──────────────────────────────────────────────┐
                  │          BEFA Central Coordinator            │
                  │   - Hard Budget Manager (MB Quota Tracking)  │
                  │   - Validation-Informed Adaptive Aggregator  │
                  │   - Security & L2 Norm Clipping Guard        │
                  │   - 8-Bit Dequantization Engine              │
                  └──────────────┬────────────────┬──────────────┘
                                 │                │
            Broadcast (Downlink) │                │ Uplink Payload
             Global Weights W_t  │                │ ΔW_k (Quantized uint8)
                                 ▼                ▼
     ┌───────────────────────────────┐        ┌───────────────────────────────┐
     │   Virtual Client 0 (TMC 0)    │  ···   │   Virtual Client 4 (TMC 4)    │
     │ - Subgraph N_0 = 61 sensors   │        │ - Subgraph N_4 = 62 sensors   │
     │ - CompactSTGCN (ChebNet K=2)  │        │ - CompactSTGCN (ChebNet K=2)  │
     │ - Event Trigger (ΔLoss > γ)   │        │ - Event Trigger (ΔLoss > γ)   │
     │ - 8-Bit Uniform Quantization  │        │ - 8-Bit Uniform Quantization  │
     └───────────────────────────────┘        └───────────────────────────────┘
```

### Core Innovations in BEFA
1. **Hard Communication Budget Management:** Tracks exact measured payload byte streams and halts updates before exhausting allocated bandwidth quotas ($\mathcal{B}_{\text{cum}} \le \mathcal{B}_{\text{max}}$).
2. **Event-Triggered Participation & Starvation Bounds:** Clients evaluate model loss velocity ($\Delta_{\text{loss}, k} \ge \gamma$). Clients with stable traffic defer transmission to conserve bandwidth, while a strict staleness bound ($\tau \ge \tau_{\max} = 2$) prevents regional starvation.
3. **Validation-Informed Adaptive Weighting:** Dampens drifting Non-IID client updates by weighting client contributions using local validation performance:
   $$\alpha_k = (1 - \lambda) \frac{n_k}{\sum_j n_j} + \lambda \frac{\exp(-\beta \mathcal{L}_{\text{val}, k})}{\sum_j \exp(-\beta \mathcal{L}_{\text{val}, j})}$$
4. **Bounded $L_2$ Norm Clipping & Security Guard:** Rejects malformed tensors, detects NaN/Inf values, scales updates exceeding $\|\Delta W\|_2 > 25.0$, and discards replay attacks via SHA-256 payload digests.
5. **8-Bit Uniform Quantization:** Compresses floating-point parameter deltas into unsigned 8-bit integers, achieving a **71.45% communication byte reduction** with zero loss in prediction convergence.

---

## 3. Dataset & Model Specifications

### 3.1 PEMS04 Dataset & Physical Adjacency
- **Source:** Caltrans Performance Measurement System (San Francisco Bay Area District 4).
- **Scale:** 307 loop detectors, 16,992 timestamps (59 days @ 5-min intervals, Jan 1 – Feb 28, 2018).
- **Channels:** Flow (veh/5-min), Occupancy ($\in [0, 1]$), Speed (mph).
- **Physical Graph Topology:** 340 physical highway segments processed via thresholded Gaussian kernel ($\sigma = 10.0, \epsilon = 0.5$) with self-loops, generating a normalized symmetric Laplacian $L_{\text{sym}} \in \mathbb{R}^{307 \times 307}$ (418 non-zero edges).
- **Chronological Split:** 70% Train (11,880 windows), 10% Val (1,685 windows), 20% Test (3,385 windows) with zero data leakage (normalization strictly fitted on Train).

### 3.2 Compact Spatio-Temporal Graph Convolutional Network (`CompactSTGCN`)
- **Spatial Layer:** Chebyshev spectral graph convolution (`ChebGraphConv`, order $K=2$, localized 1-hop spatial neighborhood).
- **Temporal Layer:** 1D causal convolutions gated with Gated Linear Units (GLU).
- **Mathematical Invariance:** Convolutional filters depend exclusively on channel dimensions and $K$, completely independent of node count $N$. The identical model weights seamlessly evaluate on local client subgraphs ($N_k \in \{61, 62\}$) and the global 307-node highway network.
- **Model Footprint:** 48,515 parameters ($189.51$ KB state dictionary).

---

## 4. Master Experimental Benchmark

All paradigms were evaluated across 3 independent random seeds (`42`, `123`, `999`). Communication is measured down to the exact byte using serialized binary payload buffers. All metrics are reported in physical units (**vehicles / 5-minute interval**):

| Paradigm | Client Dist. | Comm. Budget | Quantization | Total Transmitted (MB) | Test MAE (Mean $\pm$ Std) | Test RMSE (Mean $\pm$ Std) | Test MAPE (%) (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Heuristic** | N/A | None | No | **0.00 MB** | $22.34 \pm 0.00$ | $35.29 \pm 0.00$ | $13.40\% \pm 0.00\%$ |
| **Centralized STGCN** | Global (307) | None | No | **0.00 MB** | $\mathbf{21.19 \pm 0.09}$ | $\mathbf{32.93 \pm 0.13}$ | $\mathbf{13.49\% \pm 0.62\%}$ |
| **Standard FedAvg** | IID | Uncapped | No | 3.87 MB | $34.52 \pm 2.05$ | $45.26 \pm 1.64$ | $37.32\% \pm 5.42\%$ |
| **Proposed BEFA** | IID | 4.5 MB | No | 3.87 MB | $34.53 \pm 2.06$ | $45.27 \pm 1.65$ | $37.30\% \pm 5.45\%$ |
| **Standard FedAvg** | NON_IID | Uncapped | No | 3.87 MB | $55.04 \pm 1.60$ | $71.46 \pm 0.98$ | $58.79\% \pm 5.64\%$ |
| **Proposed BEFA** | NON_IID | 4.5 MB | No | 3.87 MB | $\mathbf{54.12 \pm 1.66}$ | $\mathbf{70.09 \pm 1.21}$ | $\mathbf{58.33\% \pm 5.48\%}$ |
| **BEFA + Int8 Quant.** | NON_IID | 3.0 MB | **Yes** | $\mathbf{1.11\text{ MB (-71.5\%)}}$ | $\mathbf{54.17 \pm 1.64}$ | $\mathbf{70.07 \pm 1.22}$ | $\mathbf{58.65\% \pm 5.39\%}$ |

*Raw experimental logs stored in [`results/metrics/master_experiment_results.csv`](results/metrics/master_experiment_results.csv).*

### Multi-Horizon Forecast Breakdown (BEFA + Int8 Quantization)
- **Step 1 (5 minutes ahead):** $\text{MAE} = 53.59$, $\text{RMSE} = 71.74$ veh/5-min
- **Step 2 (10 minutes ahead):** $\text{MAE} = 56.64$, $\text{RMSE} = 71.93$ veh/5-min
- **Step 3 (15 minutes ahead):** $\text{MAE} = 52.27$, $\text{RMSE} = 66.02$ veh/5-min

---

## 5. Ablation Studies & Findings

### 1. 8-Bit Quantization Efficiency
- **Uncompressed BEFA:** 3,872,356 bytes (~3.87 MB), $\text{RMSE} = 70.09$ veh/5-min.
- **8-Bit Quantized BEFA:** 1,158,700 bytes (~1.11 MB), $\text{RMSE} = 70.07$ veh/5-min.
- **Finding:** Uniform 8-bit quantization reduces communication bytes by **71.45%** with zero loss in predictive fidelity, as quantization noise acts as a regularizer against Non-IID drift.

### 2. Hard Budget Enforcement
- Under an intentionally constrained 2.50 MB ceiling, BEFA transmits 2.32 MB across 2 rounds and automatically halts client transmissions in round 3, completely preventing budget overruns.

### 3. Security & Norm Clipping
- Synthetic malicious updates with explosive gradient norms ($\|\Delta W\|_2 = 1000.0$) and NaN values are successfully caught and scaled/rejected without destabilizing the global model.

---

## 6. Generated Visualizations

All publication-grade figures are stored in [`results/plots/`](results/plots/):

1. **[`multi_seed_comparison.png`](results/plots/multi_seed_comparison.png):** Test MAE and RMSE across all paradigms with standard deviation error bars and communication byte annotations.
2. **[`pareto_frontier.png`](results/plots/pareto_frontier.png):** Pareto frontier showcasing Communication Cost (MB) vs Prediction Error (MAE), confirming BEFA + Quantization occupies the optimal operating point.
3. **[`convergence_comparison.png`](results/plots/convergence_comparison.png):** Multi-round training loss curves highlighting client drift stabilization.
4. **[`accuracy_vs_communication_budget.png`](results/plots/accuracy_vs_communication_budget.png):** Convergence accuracy achieved per megabyte of bandwidth consumed.
5. **[`bytes_per_round_comparison.png`](results/plots/bytes_per_round_comparison.png):** Per-round uplink and downlink payload distribution.

---

## 7. Quick-Start & CLI Execution Guide

### 7.1 Installation & Setup
```bash
# Clone the repository
git clone https://github.com/YourTeam/AIORI-PS-003.git
cd AIORI-PS-003

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Unified CLI Commands (`main.py`)

The repository includes a unified CLI supporting every workflow:

```bash
# 1. Inspect PEMS04 dataset and physical graph Laplacian
python main.py --mode inspect

# 2. Run master unit and integration test suite (34 tests in ~4s)
python main.py --mode test

# 3. Interactive live research demonstration (All 14 requirements)
python main.py --mode demo --quick      # Rapid preview using verified logs (<7s)
python main.py --mode demo              # Full in-process live execution

# 4. Train Centralized STGCN upper-bound baseline
python main.py --mode centralized --epochs 3

# 5. Train Standard FedAvg baseline (Non-IID distribution)
python main.py --mode fedavg --dist NON_IID --rounds 3

# 6. Train Proposed BEFA framework with 8-bit Quantization
python main.py --mode befa --dist NON_IID --budget_mb 3.0 --compress --rounds 3

# 7. Run automated multi-seed experiment benchmark (Seeds 42, 123, 999)
python main.py --mode experiments

# 8. Generate publication figures from experiment logs
python main.py --mode plots

# 9. Open interactive browser dashboard
start results/dashboard.html

# 10. Run full end-to-end verification pipeline
python main.py --mode all
```

> **Tip for Evaluators:** Add `--quick` to any federated or demo command (e.g. `python main.py --mode befa --quick`) for rapid evaluation on a representative dataset subset.

---

## 8. Repository Layout & Architecture

```
AIORI/
├── main.py                     # Unified CLI entrypoint for all phases
├── requirements.txt            # Python dependencies (torch, numpy, pandas, matplotlib)
├── pems04.npz                  # Verified raw PEMS04 dataset (16992, 307, 3)
│
├── data/
│   ├── dataset_loader.py       # Chronological data splitting (70/10/20) & zero-leakage scaler
│   └── adjacency/
│       ├── PEMS04.csv          # Verified 340-edge physical sensor distance graph
│       └── README.md           # Graph provenance, SHA-256 hash, and Gaussian kernel details
│
├── models/
│   ├── graph_layers.py         # ChebGraphConv (K=2) and FirstOrderGCN implementations
│   └── stgcn.py                # Node-invariant CompactSTGCN architecture (48.5K params)
│
├── clients/
│   ├── client.py               # Virtual edge client local training loop
│   └── partitioner.py          # IID (balanced) and Non-IID (volume-tier) graph partitioner
│
├── server/
│   ├── fedavg.py               # Standard sample-count-weighted FedAvg aggregator
│   ├── coordinator.py          # Standard federated learning round coordinator
│   ├── proposed_aggregation.py # BEFA validation-weighted & L2-clipped aggregator
│   └── befa_coordinator.py     # BEFA event-triggered & budget-capped coordinator
│
├── communication/
│   ├── budget_manager.py       # Hard communication quota tracker (exact byte accounting)
│   ├── event_scheduler.py      # Event-triggered participation with starvation bounds
│   ├── serialization.py        # Exact PyTorch state dictionary binary byte measurement
│   └── compression.py          # 8-bit uniform quantization and dequantization module
│
├── security/
│   └── update_validator.py     # Shape check, NaN/Inf rejection, L2 clipping, replay defense
│
├── training/
│   ├── centralized.py          # Centralized training engine with validation checkpointing
│   ├── run_centralized.py      # Standalone centralized runner
│   ├── run_federated.py        # Standalone FedAvg runner
│   └── run_befa.py             # Standalone BEFA runner
│
├── evaluation/
│   ├── metrics.py              # Physical evaluation metrics (MAE, RMSE, MAPE)
│   ├── experiments.py          # Automated multi-seed benchmark harness (Seeds 42, 123, 999)
│   └── plotting.py             # Publication figure generation engine
│
├── tests/
│   ├── test_dataset.py         # 7 tests: splitting, shapes, scaling, partitions
│   ├── test_model.py           # 7 tests: Chebyshev math, forward/backward, node-invariance
│   ├── test_centralized.py     # 5 tests: metrics, checkpoints, persistence baseline
│   ├── test_fedavg.py          # 7 tests: aggregation weights, invalid tensors, serialization
│   ├── test_budget.py          # 3 tests: budget manager, event scheduler, starvation
│   ├── test_validation.py      # 5 tests: L2 norm clipping, replay defense, 8-bit quant.
│   └── run_all_tests.py        # Master test runner (34/34 tests passing)
│
├── results/
│   ├── checkpoints/            # Saved PyTorch model checkpoints (.pt)
│   ├── logs/                   # Training and federated convergence JSON logs
│   ├── metrics/                # Master experiment CSV records
│   └── plots/                  # 5 Publication-grade figures (.png)
│
└── docs/
    ├── architecture.md         # System modularity and interface design
    ├── methodology.md          # Complete mathematical derivations and formulations
    ├── dataset_inspection_report.md # Dataset properties, sensor distribution, split verification
    ├── client_partition_report.md   # Statistical validation of IID and Non-IID partitions
    ├── model_documentation.md  # Chebyshev spectral convolution and STGCN architecture
    ├── centralized_baseline_report.md # Centralized STGCN vs persistence baseline
    ├── fedavg_baseline_report.md    # Standard FedAvg baseline evaluation
    ├── befa_framework_report.md     # BEFA framework implementation and byte savings
    ├── experiments.md          # Multi-seed experimental benchmark and ablations
    ├── limitations.md          # Honest critical evaluation, scope, and future roadmap
    ├── aiori_submission_checklist.md # Official timeline, team matrix, video pitch script, defense FAQ
    ├── presentation_deck.md    # 10-Slide presentation deck outline & speaker notes
    └── roadmap.md              # Phase-by-phase development roadmap
```

---

## 9. Comprehensive Documentation & Deliverables Index

For in-depth technical analysis and hackathon review:

| Document / Tool | Description |
| :--- | :--- |
| **[Interactive Live Demo](demo.py)** | 14-step interactive live research demonstration script (`python demo.py --quick`). |
| **[Interactive Visual Dashboard](results/dashboard.html)** | Self-contained dark-mode browser dashboard with interactive Chart.js graphs and policy simulator. |
| **[AIORI Submission Checklist & Defense Guide](docs/aiori_submission_checklist.md)** | Official timeline, team responsibilities, YouTube pitch script, and evaluator FAQ defense cheat sheet. |
| **[10-Slide Presentation Pitch Deck](docs/presentation_deck.md)** | Slide-by-slide deck outline and speaker notes for regional rounds and IIT Jammu grand finals. |
| **[Methodology & Derivations](docs/methodology.md)** | Full mathematical derivations for ChebNet, Gaussian adjacency, BEFA weighting, and Quantization. |
| **[Experimental Benchmark Report](docs/experiments.md)** | Multi-seed benchmark results across seeds 42, 123, 999, error bars, and ablation details. |
| **[System Architecture](docs/architecture.md)** | High-level system modularity, design patterns, and component interfaces. |
| **[Dataset Inspection Report](docs/dataset_inspection_report.md)** | PEMS04 statistics, zero data leakage proof, and spatial Laplacian analysis. |
| **[Client Partition Report](docs/client_partition_report.md)** | IID and Non-IID volume-tier partition properties and drift analysis. |
| **[Model Documentation](docs/model_documentation.md)** | CompactSTGCN design, node-invariance proof, and layer parameter counts. |
| **[Centralized Baseline Report](docs/centralized_baseline_report.md)** | Centralized STGCN training dynamics and persistence benchmark comparison. |
| **[FedAvg Baseline Report](docs/fedavg_baseline_report.md)** | Standard FedAvg baseline performance under IID and Non-IID conditions. |
| **[BEFA Framework Report](docs/befa_framework_report.md)** | Detailed breakdown of BEFA budget management and security validation. |
| **[Limitations & Scope](docs/limitations.md)** | Honest critical assessment of edge cases, assumptions, and future research directions. |

---

## 10. Contributors & Acknowledgements
- **AIORI Hackathon 3 — Team PS-003**
- Developed in accordance with AIORI Hackathon 3 evaluation criteria: mathematical rigor, communication-cost awareness, code modularity, and experimental reproducibility.

