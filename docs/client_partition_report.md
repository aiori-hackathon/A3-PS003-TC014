# Federated Client Partition Report: IID vs Non-IID Distributions

**Date:** 2026-10-30  
**Project:** Secure, Budgeted, Event-Aware Federated Traffic Learning  
**Virtual Clients:** 5 Simulated Traffic Clients  
**Total Sensors:** 307 Loop Detectors  

---

## 1. Overview of Partitioning Methodology
In federated traffic prediction, 5 simulated clients participate in training. To rigorously test the robustness of Federated Averaging (FedAvg) and our proposed **Budget-Aware Event-Triggered Federated Aggregation (BEFA)**, we establish two distinct distributional regimes:
1. **IID (Stratified Balanced Distribution):** Simulates homogeneous clients observing comparable traffic density regimes across the Bay Area network.
2. **Non-IID (Volume-Tier Clustering):** Simulates heterogeneous road classifications (e.g., quiet residential collectors vs. high-throughput freeway interchanges).

---

## 2. Quantitative Partition Comparison

### A. IID Partition (Stratified Balanced)
Sensors are ordered by mean traffic flow and assigned cyclically via serpentine stratification:

| Client ID | Sensor Count | Mean Flow (veh/5-min) | Std Dev Flow | Min Sensor Mean | Max Sensor Mean | Road Profile Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Client 0** | 62 | 211.49 | 93.56 | 35.71 | 467.87 | Balanced cross-section |
| **Client 1** | 62 | 212.94 | 93.80 | 47.53 | 469.70 | Balanced cross-section |
| **Client 2** | 61 | 209.87 | 88.97 | 49.68 | 444.69 | Balanced cross-section |
| **Client 3** | 61 | 211.38 | 89.15 | 57.80 | 444.83 | Balanced cross-section |
| **Client 4** | 61 | 212.81 | 90.16 | 61.12 | 448.20 | Balanced cross-section |

- **Distributional Variance Across Clients:** $< 1.5\%$ variance in mean flow ($209.87$ to $212.94$).
- **Statistical Property:** $P(X_1) \approx P(X_2) \approx \dots \approx P(X_5)$, satisfying IID assumptions.

---

### B. Non-IID Partition (Volume-Tier Heterogeneity)
Sensors are sorted by mean traffic flow and partitioned into 5 contiguous quintiles:

| Client ID | Sensor Count | Mean Flow (veh/5-min) | Std Dev Flow | Flow Range | Operational Simulation Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Client 0 (Tier 0)** | 62 | 96.76 | 22.28 | [35.71, 120.29] | Low-volume residential / minor feeder routes |
| **Client 1 (Tier 1)** | 62 | 148.17 | 19.08 | [120.29, 181.34] | Moderate suburban / secondary arterials |
| **Client 2 (Tier 2)** | 61 | 209.67 | 15.78 | [182.19, 235.09] | Urban arterials / medium transit corridors |
| **Client 3 (Tier 3)** | 61 | 260.46 | 16.71 | [236.50, 291.34] | Heavy commercial routes / bypass roads |
| **Client 4 (Tier 4)** | 61 | 346.37 | 48.84 | [293.98, 469.70] | Major highway interchanges & bottleneck junctions |

- **Distributional Divergence:** Client 4 observes **3.58x** the traffic volume of Client 0 ($346.37$ vs. $96.76$ veh/5-min).
- **Statistical Property:** Severe label and feature skew ($P_k(X) \ne P_j(X)$), creating real-world client drift.

---

## 3. Subgraph Topology and Dimension Invariance
1. Each client $k$ maintains its local induced subgraph adjacency $A^{(k)} \in \mathbb{R}^{N_k \times N_k}$ where $N_k \in \{61, 62\}$.
2. Because STGCN utilizes weight-sharing across nodes in both temporal 1D convolutions and spatial graph convolutions, client local parameter tensors have **identical shapes**:
   - Model parameter dimensions match identically across all clients.
   - Server aggregation via sample-weighted FedAvg is mathematically exact:
     $$W_{\text{global}} = \sum_{k=1}^K \frac{n_k}{\sum_j n_j} W_k$$
3. The aggregated global model can forecast either on local partitions or evaluate on the complete 307-sensor Bay Area network.
