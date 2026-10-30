# Phase 1: PEMS04 Dataset Inspection Report

**Date:** 2026-10-30  
**Project:** Secure, Budgeted, Event-Aware Federated Traffic Learning  
**Problem Statement:** PS-003  

---

## 1. Dataset File Verification
- **File Path:** `d:/AIORI/pems04.npz`
- **File Size:** 32,956,284 bytes (31.43 MB)
- **Container Format:** NumPy uncompressed/compressed `.npz` archive
- **Internal Array Keys:** `['data']`
- **Array Shape:** `(16992, 307, 3)`
- **Data Type:** `float64` (cast to `float32` for PyTorch deep learning)
- **Temporal Resolution:** 5-minute sampling rate
- **Observation Span:** 16,992 time steps = 59 days (January 1, 2018 to February 28, 2018)
- **Spatial Resolution:** 307 loop detector stations (District 4 - San Francisco Bay Area)

---

## 2. Feature Statistics & Sanity Checks

| Feature Index | Feature Name | Units | Min | Max | Mean | Std Dev | Zero Count (%) | NaN Count | Inf Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Traffic Flow** | Vehicles / 5 min | 0.00 | 919.00 | 211.7008 | 158.0684 | 82,935 (1.59%) | 0 | 0 |
| **1** | **Occupancy** | Fraction $[0, 1]$ | 0.00 | 0.7716 | 0.0528 | 0.0495 | 83,044 (1.59%) | 0 | 0 |
| **2** | **Speed** | Miles per hour | 3.00 | 85.20 | 63.4706 | 8.3557 | 0 (0.00%) | 0 | 0 |

### Observations:
1. **Zero Flow & Occupancy (1.59%):** Zero flow and occupancy align during late-night off-peak periods or brief sensor re-calibrations. There are no NaN or Inf values in the archive.
2. **Speed Distribution:** Speeds range between 3.0 mph and 85.2 mph with mean 63.47 mph, consistent with California highway free-flow speed limits and congestion dips.
3. **Target Variable:** Traffic Flow (Channel 0) will be the primary prediction target for our STGCN model.

---

## 3. Sensor Adjacency Matrix Verification
- **Source:** Standard Caltrans District 4 sensor topology (`PEMS04.csv`) from the peer-reviewed ASTGCN / ASTGNN benchmark suite.
- **Local File Path:** `data/adjacency/PEMS04.csv`
- **File SHA-256:** `68ffaa80447eb34428a9999dc28363c61406ec0fd714686d0582d2386ac6f5f7`
- **Graph Dimensions:** 307 nodes (indices 0 to 306), matching the traffic signal array exactly.
- **Raw Edges:** 340 directed road segments.
- **Distance Metrics:** Min = 3.20 m, Max = 2,712.10 m, Mean = 410.30 m, Std = 257.14 m.
- **Normalization:** Thresholded Gaussian kernel ($\sigma = 257.14$, $\kappa = 0.1$) with added self-loops ($I_{307}$).
- **Normalized Adjacency Matrix ($A$):** 983 non-zero symmetric entries (density = 1.0430%).
- **Normalized Graph Laplacian ($L_{\text{sym}}$):** $L_{\text{sym}} = I_N - D^{-1/2} A D^{-1/2}$, symmetric and positive semi-definite.

---

## 4. Chronological Splitting & Leakage Prevention
To reflect realistic operational deployment, data is split strictly chronologically:
- **Training Set (70%):** Timesteps `0` to `11,893` (11,894 steps $\approx$ 41.3 days)
- **Validation Set (10%):** Timesteps `11,894` to `13,592` (1,699 steps $\approx$ 5.9 days)
- **Testing Set (20%):** Timesteps `13,593` to `16,991` (3,399 steps $\approx$ 11.8 days)

### Anti-Leakage Guarantee:
Normalization parameters ($\mu, \sigma$) are computed **strictly from the training partition**:
- $\mu_{\text{flow}} = 207.28$, $\sigma_{\text{flow}} = 156.41$
The validation and test sets are normalized using these frozen training statistics. Model predictions are transformed back using the training parameters for final MAE and RMSE evaluation in original physical units (vehicles / 5 min).

---

## 5. Forecasting Configuration
- **Input Window ($T_{\text{in}}$):** 12 steps (60 minutes historical observations).
- **Forecast Horizon ($T_{\text{out}}$):** 3 steps (15 minutes ahead forecasting), configurable up to 12 steps (60 minutes).
- **Sliding Window Samples:**
  - Training Samples: 11,880
  - Validation Samples: 1,685
  - Testing Samples: 3,385
