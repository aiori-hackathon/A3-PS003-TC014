# AIORI PS-003: Prioritized Implementation Roadmap & Risk Checklist

**Project:** Secure, Budgeted, Event-Aware Federated Traffic Learning  
**Target Milestone:** Full Research Prototype & Experimental Validation  

---

## 1. Development Phases & Current Status

| Phase | Title | Core Tasks | Verification Milestone | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 0** | Workspace & Environment | Folder inspection, hardware check, PyTorch setup, architecture document | PyTorch installed, folder structure in place | **COMPLETE** |
| **Phase 1** | Dataset & Graph Validation | NPZ loading, adjacency provenance, chronological split, leak-free scaler, IID & Non-IID partitions | All 7 unit tests in `test_dataset.py` PASS | **COMPLETE** |
| **Phase 2** | Compact STGCN Model | Spatial GCN, Temporal Conv, residual blocks, dimension invariance, forward/backward pass | Unit tests `test_model.py` pass; finite grads | **COMPLETE** |
| **Phase 3** | Centralized Baseline | Centralized STGCN training, convergence curves, baseline MAE & RMSE evaluation | Centralized model checkpoint & logs saved | **COMPLETE** |
| **Phase 4** | Standard FedAvg Baseline | 5 Virtual Clients, sample-weighted FedAvg, IID vs Non-IID rounds, communication byte baseline | Unit tests `test_fedavg.py` pass; FedAvg metrics logged | **COMPLETE** |
| **Phase 5** | Proposed BEFA Framework | Event-triggered scheduler, hard budget enforcement, serialized byte accounting, update validation, BEFA aggregation | Unit tests `test_budget.py` & `test_validation.py` pass | **COMPLETE** |
| **Phase 6** | Comparative Experiments | 4 Paradigms (Persistence, Centralized, FedAvg, BEFA) across 3 random seeds under IID and Non-IID | Master CSV, 5 publication figures, error bars | **COMPLETE** |
| **Phase 7** | Documentation & Deliverables | Methodology report, experimental analysis, README, demonstration scripts | Submission-ready code and presentation materials | **COMPLETE** |

---

## 2. Risk Matrix & Mitigations

| Risk / Unresolved Issue | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Graph Disconnectedness in PEMS04** | Sparse adjacency matrix contains 12 graph components. | Applied self-loops ($I_{307}$) and symmetric normalized Laplacian $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$, guaranteeing stable spectral graph convolutions without NaN propagation. |
| **Data Leakage in Scaling** | Skewed evaluation or artificially high test metrics. | Scaler strictly fits $\mu, \sigma$ on the training partition ($0 \le t < 11894$). Inverse transforms applied before metric reporting. Verified by Unit Test 05. |
| **Model Weight Shape Mismatch Across Clients** | Client subgraphs have differing node counts (61 vs 62). | STGCN spatial and temporal convolution kernels are node-invariant (weights depend only on channel dimensions and Chebyshev order $K$, not node count $N$). All parameter dictionaries have identical keys and shapes. |
| **Budget Exhaustion Mid-Training** | Model stops converging prematurely if budget is poorly calibrated. | Configurable communication budget quotas with graceful deferral policy: when remaining budget is insufficient, clients defer transmission, preserve local momentum, and server keeps last verified global model. |
| **Client Starvation under Event Triggers** | Quiet sensors never trigger updates, causing local drift. | Implemented maximum update interval threshold $\tau_{\max}$. If a client has not communicated for $\tau_{\max}$ rounds, participation is forced regardless of event score. |
| **Hardware Constraints on Windows / CPU vs GPU** | Training runtime could be long on CPU. | PyTorch 2.15 is installed. The compact STGCN model is deliberately dimensioned for fast training (hidden channels $\le 32$, batch size 32-64, subsampled epochs or early stopping for prototyping). CUDA support is readily available if GPU wheel is activated. |
