# System Scope, Limitations, and Future Roadmap
**Project:** Secure, Budgeted, Event-Aware Federated Traffic Learning (BEFA)  
**Hackathon:** AIORI Hackathon 3 – Problem Statement PS-003  
**Date:** October 2026

---

## 1. System Scope & Boundaries

In strict compliance with the AIORI Hackathon 3 problem guidelines:
- **Scope Definition:** This project is a rigorous, research-grade software prototype designed to validate the algorithmic feasibility, mathematical convergence, and communication-cost dynamics of federated spatio-temporal learning.
- **Excluded Dependencies:** In accordance with competition rules, this prototype explicitly does **not** interface with live CCTV video feeds, physical loop detector hardware, embedded microcontroller boards, or municipal traffic light controllers.

---

## 2. Dataset & Topological Limitations

### 2.1 Highway vs. Urban Grid Traffic
- **Freeway Characteristic:** The PEMS04 dataset records California state highway loop detectors (interstates I-80, I-880, US-101, SR-24, etc.). Traffic flow on freeways is largely continuous and governed by macroscopic speed-density-flow relationships (Green-shield's or Underwood's models).
- **Urban Generalization:** Urban road networks (such as Manhattan or downtown Singapore) are heavily influenced by discrete signal phases, pedestrian crossings, and grid bottlenecks. Extending `CompactSTGCN` to urban grids would require augmenting the adjacency matrix with directed edge types, lane configurations, and signal timing phase features.

### 2.2 Graph Connectivity Properties
- **Empirical Graph Structure:** The 340-edge distance graph for the 307 sensors contains 12 connected components when thresholded by physical road distance ($\epsilon = 0.5$). While the largest component encompasses 237 sensors, several peripheral freeway spurs form isolated sub-graphs.
- **Handling in BEFA:** The Chebyshev polynomial formulation ($K=2$) and self-loop inclusion ensure numerical stability and localized 1-hop spatial diffusion across both the giant component and isolated subgraphs.

---

## 3. Federated Simulation Limitations

### 3.1 In-Process Virtual Client Execution
- **Host Execution:** The 5 virtual clients operate sequentially or concurrently on a single host machine, coordinating via the centralized `BefaCoordinator`.
- **Payload Fidelity:** While physical wireless radio links are not instantiated, payload measurement is **100% exact**: updates are serialized into binary `torch.save` memory streams, converted to quantized integer byte-buffers, and measured using exact `sys.getsizeof` / buffer length tracking. There are zero synthetic or interpolated byte estimates.
- **Network Realism:** The current prototype does not model packet loss, TCP retransmission overhead, cellular 5G latency jitter, or asymmetric uplink/downlink bandwidth limits.

### 3.2 Synchronous Round Coordination
- **Coordination Model:** BEFA employs synchronous round-based aggregation where participating clients synchronize at each global step.
- **Straggler Risk in Real Deployments:** In real-world edge networks with heterogeneous compute hardware, slow clients (stragglers) could delay global aggregation. While the event-triggered scheduler reduces the number of active uploaders, an asynchronous federated protocol (e.g., FedAsync or FedBuff) would be necessary for large-scale production deployments.

---

## 4. Security & Privacy Boundaries

### 4.1 Lightweight Validation vs. Cryptographic Security
- **Implemented Defenses:**
  - Strict input tensor shape verification.
  - Zero-tolerance NaN / Inf rejection.
  - Bounded $L_2$ norm clipping ($\|\Delta W\|_2 \le 25.0$) against gradient explosion and poisoning attacks.
  - SHA-256 payload digest tracking against replay attacks.
  - Staleness-aware validation weighting.
- **Omitted Cryptographic Mechanisms:**
  - **Differential Privacy (DP):** Formal Gaussian or Laplacian noise injection ($\epsilon, \delta$-DP) is omitted in this baseline to preserve baseline accuracy comparisons without gradient degradation.
  - **Secure Multi-Party Computation (SMPC):** Cryptographic secret sharing is not implemented, meaning the server observes unencrypted client parameter updates.
  - **Homomorphic Encryption (HE):** Fully homomorphic encryption is computationally intractable for multi-layer neural network parameters on resource-constrained edge devices.

---

## 5. Scalability & Computational Constraints

- **Parameter Efficiency:** The `CompactSTGCN` model was intentionally engineered with 48,515 parameters (~189.5 KB uncompressed), rendering it well-suited for CPU and edge execution.
- **Scaling to Giant Networks:** For ultra-large networks such as PEMS07 (883 sensors) or California statewide graphs ($N > 10,000$), full Chebyshev matrix multiplications $T_k(\tilde{L}) X$ require sparse tensor optimizations (`torch.sparse`) or mini-batch sub-graph sampling (e.g., GraphSAINT or Cluster-GCN).

---

## 6. Strategic Future Roadmap

| Milestone | Objective | Key Innovation |
| :--- | :--- | :--- |
| **Phase 8 (Async)** | Asynchronous Client Updates | Implement buffer-based aggregation to eliminate synchronization barriers. |
| **Phase 9 (Privacy)** | Rényi Differential Privacy (RDP) | Incorporate adaptive client-side DP clipping with privacy budget accounting. |
| **Phase 10 (Dynamic Graph)** | Graph Structure Learning | Learn dynamic time-varying adjacency matrices $\tilde{A}_t$ via spatial self-attention. |
| **Phase 11 (Field Testing)** | Embedded Edge Validation | Deploy client containers to Raspberry Pi 5 / NVIDIA Jetson Orin edge nodes. |
