# Phase 4: Standard FedAvg Baseline Report

**Project:** Secure, Budgeted, Event-Aware Federated Traffic Learning (BEFA)  
**Dataset:** Full PEMS04 (11,880 train windows, 1,685 val windows, 3,385 test windows)  
**Model:** `CompactSTGCN` (48,515 parameters, 189.51 KB uncompressed)  
**Number of Virtual Clients:** 5 Simulated Traffic Clients  
**Aggregation Strategy:** Standard Sample-Count-Weighted Federated Averaging (McMahan et al.)  
**Date:** 2026-10-30  

---

## 1. Experimental Methodology
- **Rounds:** 3 communication rounds
- **Client Local Training:** 1 local epoch per round (Adam, $\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$, batch size $= 128$)
- **Communication Accounting:** Exact serialized payload measurement (`torch.save` to `io.BytesIO`)
  - Download bytes per client per round: $202,895$ bytes $\approx 198.14$ KB
  - Upload bytes per client per round: $202,895$ bytes $\approx 198.14$ KB
  - Total bytes per round (5 clients): $2,028,950$ bytes $\approx 1,981.4$ KB $\approx 1.93$ MB
- **Experimental Distributions:**
  1. **IID:** Stratified balanced distribution (cross-client flow variance $< 1.5\%$)
  2. **Non-IID:** Volume-tier clustering (Tier 0 mean flow $95.1$ to Tier 4 mean flow $342.3$, $3.6\times$ skew)

---

## 2. Quantitative Results Comparison: IID vs. Non-IID FedAvg

| Metric | Centralized Baseline | Standard FedAvg (IID) | Standard FedAvg (Non-IID) | Impact of Non-IID Heterogeneity |
| :--- | :--- | :--- | :--- | :--- |
| **Communication Rounds** | 0 (Centralized) | 3 rounds | 3 rounds | Identical schedule |
| **Total Transmitted Bytes** | 0 bytes | **6,086,850 bytes** (5.80 MB) | **6,086,850 bytes** (5.80 MB) | Blind 100% transmission |
| **Bytes per Round** | N/A | 1,981.4 KB / round | 1,981.4 KB / round | Inflexible overhead |
| **Test MAE (Overall)** | 21.80 veh/5-min | **20.81 veh/5-min** | **43.08 veh/5-min** | $+107.0\%$ error increase |
| **Test RMSE (Overall)** | 32.70 veh/5-min | **32.07 veh/5-min** | **62.47 veh/5-min** | $+94.8\%$ error increase |
| **Test MAPE (Overall)** | 17.95 % | **14.62 %** | **35.50 %** | Significant distortion |
| **Step 1 (5m) MAE / RMSE** | 19.45 / 29.84 | 19.20 / 29.78 | 42.49 / 62.04 | Severe short-term drift |
| **Step 2 (10m) MAE / RMSE** | 22.96 / 33.07 | 20.57 / 31.94 | 43.64 / 63.87 | Divergent trajectories |
| **Step 3 (15m) MAE / RMSE** | 22.99 / 34.98 | 22.66 / 34.34 | 43.11 / 61.46 | Persistent distortion |

---

## 3. Round-by-Round Convergence Analysis

### A. IID Convergence Log (`fedavg_iid_seed42_rounds_log.csv`)
| Round | Active Clients | Train Loss | Val MAE (veh/5-min) | Val RMSE (veh/5-min) | Round KB | Cumulative MB | Round Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 5 / 5 | 0.0437 | 23.11 | 34.67 | 1,981.4 KB | 1.93 MB | 143.0 s |
| **2** | 5 / 5 | 0.0265 | 21.94 | 33.58 | 1,981.4 KB | 3.87 MB | 143.5 s |
| **3** | 5 / 5 | **0.0255** | **21.63** | **33.31** | 1,981.4 KB | **5.80 MB** | 143.6 s |

### B. Non-IID Convergence Log (`fedavg_non_iid_seed42_rounds_log.csv`)
| Round | Active Clients | Train Loss | Val MAE (veh/5-min) | Val RMSE (veh/5-min) | Round KB | Cumulative MB | Round Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 5 / 5 | 0.0510 | 47.79 | 68.82 | 1,981.4 KB | 1.93 MB | 147.6 s |
| **2** | 5 / 5 | 0.0339 | 52.63 | 69.31 | 1,981.4 KB | 3.87 MB | 152.3 s |
| **3** | 5 / 5 | **0.0318** | **45.78** | **66.14** | 1,981.4 KB | **5.80 MB** | 149.0 s |

---

## 4. Key Scientific Insights
1. **IID Convergence:** Under IID conditions, standard FedAvg achieves strong convergence, with test RMSE (32.07) matching the centralized model (32.70), demonstrating that sample-weighted averaging is mathematically sound.
2. **The Non-IID Vulnerability of Conventional FedAvg:** Under volume-tier non-IID conditions, client updates pull in diverging directions due to disparate flow regimes ($95.1$ vs $342.3$ veh/5-min). The aggregated model suffers from substantial client drift (RMSE increases from 32.07 to 62.47 veh/5-min).
3. **Communication Waste:** In standard FedAvg, communication expenditure is completely unbudgeted and static ($1.93$ MB / round). All 5 clients transmit their full model state in every round regardless of whether local conditions changed or whether an update adds marginal value.
4. **Motivation for Proposed BEFA (Phase 5):** This establishes the explicit baseline requirement for our proposed **Budget-Aware Event-Triggered Federated Aggregation (BEFA)**, which introduces:
   - Event-triggered client participation (preventing unnecessary transmissions).
   - Hard byte budget enforcement.
   - Validation-informed aggregation.
   - Update norm clipping and security safeguards.
