# AIORI Hackathon 3 (PS-003): Submission Checklist, Protocol & Team Defense Guide
**Project:** Secure, Budgeted, Event-Aware Federated Traffic Learning (BEFA)  
**Hackathon Target:** Regional Round (Nov 2026) & National Finals (Dec 17, 2026 @ IIT Jammu)  
**Status:** 100% Submission Ready

---

## 1. Official AIORI Hackathon Timeline & Milestones

| Stage | Official Window | Team Action Required | Status |
| :--- | :--- | :--- | :---: |
| **Level 1 (Initial Screening)** | Oct 14 | Project registration, problem statement PS-003 selection | Completed |
| **Level 2 (Announcement)** | Oct 16 | Team qualification confirmation | Completed |
| **Mentoring & Prototype (Steps 8–10)**| Oct 16 – Oct 31 | Prototype implementation, multi-seed experiments, documentation | **COMPLETE** |
| **Evaluator Contact & Screening** | Oct 31 – Nov 4 | Live code walkthrough, test execution, evaluator Q&A | **READY** |
| **Regional Stage** | Nov 5 / Nov 24–30 | Regional presentation & demonstration pitch | **READY** |
| **National Finals** | Dec 17 (IIT Jammu) | Grand finale defense, live prototype demo, prize judging | **READY** |

---

## 2. AIORI Technical Submission Checklist

- [x] **Official Repository Template:** Cloned and maintained using the designated AIORI template structure.
- [x] **Repository Topics:** Tagged with `aiori-hackathon` on GitHub.
- [x] **Team Code Repository:** Repository named cleanly according to team code guidelines.
- [x] **No Forbidden Dependencies:** Zero dependencies on live CCTV, traffic light hardware, or simulated municipal APIs (pure software prototype).
- [x] **Reproducible Environment:** `requirements.txt` specifies exact dependencies (`torch`, `numpy`, `pandas`, `matplotlib`).
- [x] **Unified CLI Interface:** [`main.py`](../main.py) and [`demo.py`](../demo.py) allow judges to execute any stage with single commands.
- [x] **Master Test Suite:** 34/34 tests passing in [`tests/run_all_tests.py`](../tests/run_all_tests.py) across 6 test suites.
- [x] **Multi-Seed Validation:** 3 independent random seeds (`42, 123, 999`) logged in [`results/metrics/master_experiment_results.csv`](../results/metrics/master_experiment_results.csv).
- [x] **Exact Byte Accounting:** Communication measured down to the byte via binary serialized state dictionaries (no parameter count estimation).
- [x] **Interactive Dashboard:** Self-contained HTML5/CSS3 visual dashboard in [`results/dashboard.html`](../results/dashboard.html).
- [x] **Comprehensive Documentation:** 10 detailed technical reports in [`docs/`](../docs/).

---

## 3. Team Responsibility Allocation Matrix

As required by Section 12 of the Master Specification, tasks are strictly partitioned between our two-member student engineering team:

```
               AIORI TEAM PS-003: ENGINEERING ALLOCATION
┌───────────────────────────────────────┬───────────────────────────────────────┐
│               MEMBER 1                │               MEMBER 2                │
│       Machine Learning & Modeling     │      Communication & Security         │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ • PEMS04 Data Preprocessing & Splitting│ • Exact Payload Serialization Byte Log│
│ • Zero-Leakage Data Scaler Formulation │ • Hard Budget Quota Enforcement Engine│
│ • CompactSTGCN Graph Architecture     │ • Event-Triggered Client Scheduler    │
│ • Chebyshev Spectral Conv (K=2) Math  │ • Starvation Bound Guard (tau_max=2)  │
│ • Centralized Baseline Training Engine│ • 8-Bit Uniform Quantization Engine   │
│ • Virtual Client Training Loops       │ • Update Validator & L2 Norm Clipping │
│ • Multi-Seed Experiment Automation    │ • Duplicate & Replay Rejection Defense│
│ • Physical Metric Evaluation (MAE/RMSE│ • Resource Cleanup & Memory Management│
└───────────────────────────────────────┴───────────────────────────────────────┘
                                   │
                                   ▼
                       SHARED RESPONSIBILITIES:
 • System Architecture Design & FedAvg Sample-Weighted Mathematics
 • Validation-Informed Adaptive BEFA Weighting Formulation
 • Dataset Empirical Analysis (Freeway Topology vs Urban Grids)
 • Live Prototype Demonstration (demo.py & dashboard.html)
 • AIORI Evaluator Defense & Presentation Pitch
```

---

## 4. Technical Video Presentation Pitch Script (3–4 Minutes)

*Guidance for recording the team's official YouTube submission video:*

### Part 1: Problem & Motivation (0:00 – 0:50) — Presented by Member 1
> *"Good morning, esteemed judges and evaluators. We are Team PS-003, presenting **BEFA**: Secure, Budgeted, Event-Aware Federated Traffic Learning for AIORI Hackathon 3.*  
> *Modern smart cities rely on traffic loop detectors across regional highway corridors. Centralizing this raw telemetry creates privacy risks, single-point failures, and massive bandwidth costs. While Federated Learning (FedAvg) offers a decentralized alternative, conventional FedAvg has two critical flaws in traffic management:*  
> *First, severe volume disparity between rural ramps and busy freeways causes catastrophic **client drift** under Non-IID distributions.*  
> *Second, clients upload uncompressed models every single round, exhausting cellular IoT communication budgets.*  
> *Our solution, **BEFA**, is engineered specifically to conquer both bottlenecks."*

### Part 2: Innovation & System Architecture (0:50 – 2:00) — Presented by Member 2
> *"To solve this, BEFA introduces five integrated pillars on top of a compact Spatio-Temporal Graph Convolutional Network:*  
> *1. **Node-Invariant Graph Convolutions:** Using 2nd-order Chebyshev polynomials, our model weights are strictly independent of sensor count, allowing identical weights to execute on local 61-sensor subgraphs and the global 307-sensor server.*  
> *2. **Hard Communication Budget Quotas:** Our coordinator measures exact binary serialized payload bytes, guaranteeing zero budget overruns.*  
> *3. **Event-Triggered Scheduling:** Clients only transmit when traffic loss velocity exceeds a threshold ($\gamma$), with a starvation bound ($\tau_{\max}=2$) ensuring quiet regions are never silenced.*  
> *4. **Validation-Informed Adaptive Weighting with $L_2$ Clipping:** Drifting client updates are bounded ($\|\Delta W\|_2 \le 25.0$) and weighted by empirical validation loss, mitigating non-IID divergence.*  
> *5. **8-Bit Uniform Quantization:** Float32 updates are compressed to uint8, slashing communication bytes by **71.45%** with zero loss in predictive accuracy."*

### Part 3: Experimental Validation & Live Demo (2:00 – 3:15) — Presented Jointly
> *"We validated BEFA on the California Caltrans PEMS04 dataset—16,992 timestamps across 307 sensors—using a strict chronological 70/10/20 split with zero data leakage.*  
> *Testing across 3 independent random seeds (42, 123, 999), our benchmark proves:*  
> *- Centralized STGCN achieves an upper bound MAE of 21.19 veh/5-min.*  
> *- Standard FedAvg under Non-IID skew deteriorates to an RMSE of 71.46 veh/5-min.*  
> *- Proposed BEFA stabilizes client drift, improving RMSE to 70.09 veh/5-min.*  
> *- With 8-bit quantization, BEFA transmits only **1.11 MB** across 3 rounds—saving over 2.76 MB compared to FedAvg's 3.87 MB—while matching uncompressed accuracy!*  
> *Our entire test suite of 34 unit and integration tests executes in just 4.18 seconds with a 100% pass rate."*

### Part 4: Conclusion & Impact (3:15 – 3:45)
> *"BEFA demonstrates that communication-aware federated learning can achieve superior traffic predictions within real-world bandwidth quotas. All code, checkpoints, and interactive dashboards are fully reproducible. Thank you!"*

---

## 5. Evaluator Defense Cheat Sheet (FAQ)

### Q1: Why did you choose Chebyshev Graph Convolutions ($K=2$) over standard GCN or Graph Attention (GAT)?
**Answer:**  
Standard GCN ($K=1$) is limited to immediate nearest neighbors. ChebNet with $K=2$ allows 2-hop spatial receptive fields via the recurrence $T_2(\tilde{L}) = 2\tilde{L}T_1 - T_0$, capturing upstream and downstream traffic shockwaves without requiring explicit eigendecomposition $\mathcal{O}(N^3)$. Furthermore, GAT introduces quadratic attention parameter overhead ($\mathcal{O}(N^2)$), whereas `CompactSTGCN` has only 48,515 parameters (~189.5 KB), making it lightweight for edge execution.

### Q2: Why does 8-bit quantization not degrade prediction accuracy?
**Answer:**  
Neural network parameter updates $\Delta W$ in federated learning exhibit heavy-tailed bell-shaped distributions centered around zero. Uniform 8-bit quantization introduces a maximum reconstruction error bounded by $\frac{\max - \min}{510}$, yielding an empirical Mean Squared Error $\text{MSE} < 5 \times 10^{-5}$. Under Non-IID conditions, this slight perturbation acts as stochastic noise regularization, preventing local over-fitting and yielding a test RMSE of $70.07$ vs. $70.09$ veh/5-min.

### Q3: How do you guarantee zero data leakage between Train, Val, and Test?
**Answer:**  
We enforce a strict chronological 70% Train, 10% Val, and 20% Test split. Normalization statistics ($\mu = 208.62$, $\sigma = 157.06$) are computed **strictly on the 70% training partition**. The validation and test sets are normalized using these frozen training statistics and inverse-transformed prior to calculating physical MAE and RMSE in vehicles / 5-min.

### Q4: Why did you not include Differential Privacy (DP) with Gaussian noise?
**Answer:**  
In accordance with Problem Statement PS-003 guidelines, this project prioritizes communication cost evaluation and non-IID drift mitigation. Uncalibrated differential privacy noise injection degrades spatial gradient signals in small edge models. We opted for lightweight security safeguards (shape checking, NaN rejection, bounded $L_2$ norm clipping, and SHA-256 replay detection) while highlighting Rényi Differential Privacy as a roadmap milestone in [`docs/limitations.md`](limitations.md).

### Q5: How did you measure communication bytes?
**Answer:**  
We do not estimate communication from parameter counts. Each client update is serialized into a standard binary in-memory byte buffer via `torch.save`. The exact buffer length in bytes is recorded. For quantized payloads, the byte buffer comprises the uint8 tensor, scale float32, zero-point float32, and dimension headers.
