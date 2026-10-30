# PEMS04 Sensor Distance & Adjacency Matrix Documentation

## 1. Source and Provenance
- **Dataset:** PeMS04 (Caltrans Performance Measurement System District 4 - San Francisco Bay Area).
- **Time Range:** January 1, 2018 to February 28, 2018 (59 days, 16,992 intervals of 5 minutes).
- **Sensors:** 307 loop detector stations.
- **Source Repository:** Standard ASTGCN / ASTGNN Traffic Forecasting Benchmark (`guoshnBJTU/ASTGNN`, Beijing Jiaotong University).
- **Original Edge List File:** `data/adjacency/PEMS04.csv`
- **SHA-256 Hash:** `68ffaa80447eb34428a9999dc28363c61406ec0fd714686d0582d2386ac6f5f7`
- **Edges Count:** 340 directed road connectivity pairs.
- **Columns:** `from,to,cost` where `cost` represents road network distance in meters.

## 2. Sensor Node Mapping
- The node IDs in `PEMS04.csv` span from `0` to `306`, exactly corresponding to the 307 sensors in the traffic signal tensor `pems04.npz` (shape: `(16992, 307, 3)`).
- Index mapping is direct: sensor index `i` in `pems04.npz[:, i, :]` aligns with node ID `i` in `PEMS04.csv`.

## 3. Mathematical Normalization (Adjacency Matrix & Graph Laplacian)
For Spatio-Temporal Graph Convolutional Networks (STGCN), the raw road distance matrix $W$ is converted into an adjacency matrix $A$ and normalized graph Laplacian $L$:

### Step A: Thresholded Gaussian Kernel (Distance to Proximity)
$$A_{ij} = \begin{cases} \exp\left(-\left(\frac{\text{dist}(v_i, v_j)}{\sigma}\right)^2\right), & \text{if } \text{dist}(v_i, v_j) \le \kappa \text{ and } i \ne j \\ 0, & \text{otherwise} \end{cases}$$
where $\sigma$ is the standard deviation of valid pairwise distances ($\sigma \approx \text{std}(\text{distances})$) and $\kappa$ is the threshold (or threshold $\epsilon = 0.1$).

### Step B: Symmetric Normalized Adjacency Matrix (GCN / ChebNet)
Self-loops are added: $\tilde{A} = A + I_N$.
The degree matrix is $\tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}$.
The normalized adjacency is:
$$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$

For Chebyshev polynomial approximation (ChebNet):
$$L = I_N - D^{-1/2} A D^{-1/2}$$
$$\tilde{L} = \frac{2}{\lambda_{\max}} L - I_N$$
where $\lambda_{\max} \approx 2.0$.
