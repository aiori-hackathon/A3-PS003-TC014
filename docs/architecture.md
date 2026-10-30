# System Architecture: Secure, Budgeted, Event-Aware Federated Traffic Learning (BEFA)

**Problem Statement:** PS-003  
**Target:** Spatial-Temporal Traffic Flow Forecasting under Communication-Budget Constraints  

---

## 1. High-Level Architectural Diagram

```
+-----------------------------------------------------------------------------------+
|                                  CENTRAL SERVER                                   |
|                                                                                   |
|  +--------------------+    +-----------------------+    +----------------------+  |
|  |  Global STGCN      |    |  BEFA Aggregation     |    |  Budget & Security   |  |
|  |  Model Parameters  |    |  - Sample-weighted    |    |  - Byte Accounting   |  |
|  |  (Spatial-Temporal)|    |  - Event-informed     |    |  - Stale/Dupl Check  |  |
|  +---------+----------+    +-----------^-----------+    +----------^-----------+  |
|            |                           |                           |              |
+------------|---------------------------|---------------------------|--------------+
             | Global Broadcast          | Filtered Local Updates    | Budget Quota
             v                           |                           |
+-----------------------------------------------------------------------------------+
|                             VIRTUAL CLIENTS (5 Nodes)                             |
|                                                                                   |
|  Client 0 (Tier 0)    Client 1 (Tier 1)    Client 2 (Tier 2)   ... Client 4 (Tier4)|
|  +----------------+   +----------------+   +----------------+                      |
|  | Sensors: 62    |   | Sensors: 62    |   | Sensors: 61    |                      |
|  | Local STGCN    |   | Local STGCN    |   | Local STGCN    |                      |
|  | Event Detector |   | Event Detector |   | Event Detector |                      |
|  | Serializer/Q   |   | Serializer/Q   |   | Serializer/Q   |                      |
|  +----------------+   +----------------+   +----------------+                      |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Modules and Responsibilities

### Module 1: `data/` (Dataset & Graph Topology)
- **`dataset_loader.py`**: Loads `pems04.npz`, enforces strict chronological split (70% train, 10% val, 20% test), performs leak-free scaling, generates sliding windows ($T_{\text{in}}=12 \to T_{\text{out}}=3$).
- **`adjacency/`**: Houses verified PEMS04 distance edge list (`PEMS04.csv`), generates thresholded Gaussian kernel adjacency ($A$) and symmetric normalized graph Laplacian ($L_{\text{sym}}$).

### Module 2: `models/` (Spatial-Temporal Architecture)
- **`graph_layers.py`**: First-order Chebyshev / Kipf-Welling Spatial Graph Convolution with node-invariant weight tensors.
- **`stgcn.py`**: Compact Spatio-Temporal Graph Convolutional Network:
  - Temporal Gated Linear Unit (GLU) Conv 1
  - Spatial Graph Conv
  - Temporal Gated Linear Unit (GLU) Conv 2
  - Fully convolutional temporal output projection.

### Module 3: `clients/` (Federated Nodes & Partitioning)
- **`partitioner.py`**: Allocates 307 sensors across 5 virtual clients under IID (stratified balanced) and Non-IID (volume-tier) distributions.
- **`client.py`**: Manages client local dataset, local SGD optimizer, loss computation, and local event detection.

### Module 4: `server/` (Federated Aggregators)
- **`fedavg.py`**: Standard sample-weighted McMahan FedAvg baseline:
  $$W_{\text{global}}^{(t+1)} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j} W_k^{(t+1)}$$
- **`proposed_aggregation.py`**: BEFA Aggregator incorporating event importance, parameter freshness, and update clipping.
- **`coordinator.py`**: Orchestrates round scheduling, broadcast, collection, and aggregation.

### Module 5: `communication/` (Budget & Event Triggering)
- **`budget_manager.py`**: Measures exact serialized payload bytes (download and upload) using PyTorch serialization or protocol buffers, tracks cumulative budget, enforces hard spending cap.
- **`event_scheduler.py`**: Computes local traffic change metrics ($\Delta \bar{X}_{\text{recent}}$, forecast error drift) and triggers participation only when traffic state diverges significantly. Enforces max update interval $\tau_{\max}$ to avoid client starvation.
- **`compression.py`**: Optional lightweight 8-bit quantization with scale and zero-point parameters.

### Module 6: `security/` (Safeguards & Validation)
- **`update_validator.py`**: Validates parameter shapes, finite value guarantees (no NaN/Inf), norm clipping ($L_2$ clipping), stale update rejection, duplicate update prevention.

### Module 7: `evaluation/` & `tests/`
- **`metrics.py`**: Ground-truth inverted scale MAE, RMSE, MAPE.
- **`experiments.py`**: Automated testbed across 4 paradigms: Persistence, Centralized STGCN, FedAvg, and BEFA across 3 random seeds.
