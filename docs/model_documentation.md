# Phase 2: Compact STGCN Model Documentation

**Project:** Secure, Budgeted, Event-Aware Federated Traffic Learning (BEFA)  
**Model Architecture:** Compact Spatio-Temporal Graph Convolutional Network (`CompactSTGCN`)  
**Implementation Files:** [`models/stgcn.py`](file:///d:/AIORI/models/stgcn.py), [`models/graph_layers.py`](file:///d:/AIORI/models/graph_layers.py)  

---

## 1. Architectural Motivation & Theoretical Foundation
Traffic flow exhibits strong spatial-temporal correlations:
1. **Temporal Dependencies:** Current traffic at time $t$ is strongly conditioned on historical observations $[t - T_{\text{in}} + 1, \dots, t]$.
2. **Spatial Topology:** Traffic state propagates along physical road segments dictated by network connectivity and lane geometry.

Traditional RNNs (e.g., LSTM, GRU) suffer from slow sequential execution and gradient degradation over long sequences. `CompactSTGCN` employs fully convolutional architectures:
- **Temporal Modeling:** 1D Gated Linear Unit (GLU) temporal convolutions along time dimension.
- **Spatial Modeling:** Chebyshev Spectral Graph Convolutions ($K=2$) over the normalized graph Laplacian $L_{\text{sym}}$.

---

## 2. Mathematical Formulations

### A. Temporal Gated Linear Unit Convolution (TConvGLU)
For input feature map $X \in \mathbb{R}^{B \times C_{\text{in}} \times T \times N}$, the 1D temporal convolution with kernel size $K_t = 3$ produces double the output channels:
$$\Phi(X) = [P, Q] \in \mathbb{R}^{B \times 2C_{\text{out}} \times T \times N}$$
The non-linear GLU gating operation is:
$$\text{GLU}(P, Q) = P \odot \sigma(Q) \in \mathbb{R}^{B \times C_{\text{out}} \times T \times N}$$
where $\sigma(\cdot)$ is the sigmoid gate and $\odot$ is the Hadamard element-wise product. This provides smooth gradient highways and mitigates vanishing gradients.

### B. Chebyshev Spectral Graph Convolution (ChebGraphConv)
Given scaled graph Laplacian $\tilde{L} = L_{\text{sym}} - I_N$ (assuming $\lambda_{\max} \approx 2$):
$$T_0(\tilde{L}) = I_N, \quad T_1(\tilde{L}) = \tilde{L}, \quad T_k(\tilde{L}) = 2\tilde{L} T_{k-1}(\tilde{L}) - T_{k-2}(\tilde{L})$$
The spatial graph convolution is defined as:
$$Z = \sum_{k=0}^{K-1} T_k(\tilde{L}) X \Theta_k \in \mathbb{R}^{B \times C_{\text{out}} \times T \times N}$$
where $\Theta_k \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}$ is the learnable kernel matrix.

### C. Node Invariance for Federated Aggregation
Crucially, kernel weights $\Theta_k$ and temporal 1D convolution weights do **not** depend on the number of nodes $N$:
$$\text{Weight Dimensions: } (K, C_{\text{in}}, C_{\text{out}})$$
Consequently:
- Clients with $N_k = 61$ or $62$ sensors have **identical model parameter keys and shapes** to the global server model ($N = 307$).
- Parameter updates $W_k$ can be directly averaged via sample-weighted FedAvg without tensor reshaping.

---

## 3. Layer Specifications & Parameter Count

| Component | Layer Type | Input Shape | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **ST-Conv Block 1** | TemporalConvGLU 1 | $(B, 3, 12, N)$ | $(B, 32, 12, N)$ | 640 |
| | ChebGraphConv | $(B, 32, 12, N)$ | $(B, 32, 12, N)$ | 2,080 |
| | TemporalConvGLU 2 | $(B, 32, 12, N)$ | $(B, 32, 12, N)$ | 6,208 |
| | Residual Conv | $(B, 3, 12, N)$ | $(B, 32, 12, N)$ | 128 |
| | BatchNorm2d | $(B, 32, 12, N)$ | $(B, 32, 12, N)$ | 64 |
| **ST-Conv Block 2** | TemporalConvGLU 1 | $(B, 32, 12, N)$ | $(B, 32, 12, N)$ | 6,208 |
| | ChebGraphConv | $(B, 32, 12, N)$ | $(B, 32, 12, N)$ | 2,080 |
| | TemporalConvGLU 2 | $(B, 32, 12, N)$ | $(B, 32, 12, N)$ | 6,208 |
| | BatchNorm2d | $(B, 32, 12, N)$ | $(B, 32, 12, N)$ | 64 |
| **Forecasting Head** | Temporal Conv2d | $(B, 32, 12, N)$ | $(B, 64, 1, N)$ | 24,640 |
| | Projection Conv2d | $(B, 64, 1, N)$ | $(B, 3, 1, N)$ | 195 |
| **Total** | | | | **48,515 parameters** |

- **Uncompressed Parameter Size:** $48,515 \times 4\text{ bytes} \approx 194,060\text{ bytes} = 189.51\text{ KB}$.
- **Suitability for Federated Communication:** Lightweight footprint facilitates ultra-low transmission overhead per communication round, enabling exact measurement of serialized payload bytes under tight budgets.

---

## 4. Verification & Unit Test Results (`tests/test_model.py`)
All 7 unit tests passed with 100% success:
1. `test_01_chebyshev_polynomial_math`: **PASSED** ($T_0 = I$, $T_1 = L - I$, recurrence holds).
2. `test_02_cheb_graph_conv_forward`: **PASSED** (Output shape $(B, C_{\text{out}}, T, N)$, no NaNs/Infs).
3. `test_03_dimension_invariance_across_node_counts`: **PASSED** (Executes identically on $N=307$ and $N=61$ with identical weights).
4. `test_04_forward_backward_and_optimizer_step`: **PASSED** (Finite gradients computed, optimizer updates weights).
5. `test_05_reproducible_initialization`: **PASSED** (Fixed seeds produce deterministic initial weights).
6. `test_06_parameter_count_and_footprint`: **PASSED** (48,515 parameters, 189.51 KB).
7. `test_07_real_pems04_batch_execution`: **PASSED** (End-to-end forward/backward with actual PEMS04 batch).
