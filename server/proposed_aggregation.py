"""
Proposed BEFA (Budget-Aware Event-Triggered Federated Aggregation) Module.
Formulates an intelligent, mathematically grounded aggregation strategy:
Combines sample-count weights with local event importance and validation quality scores:

w_i = (n_i / sum_j n_j) * exp(beta * min(E_i, 1.0)) * (1 / (1 + L_val,i))
W_global = sum_{i in S_t} (w_i / sum_k w_k) * W_i
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch


def compute_befa_weights(
    client_metadata: List[Dict[str, Any]],
    beta: float = 0.3
) -> np.ndarray:
    """
    Computes normalized BEFA aggregation weights across participating clients.

    Each metadata dict must contain:
      - 'num_samples': int (n_i)
      - 'event_score': float (E_i)
      - 'val_loss': float (L_val,i)
    """
    num_clients = len(client_metadata)
    total_samples = sum(m["num_samples"] for m in client_metadata)

    raw_weights = np.zeros(num_clients, dtype=np.float64)

    for i, m in enumerate(client_metadata):
        n_i = m["num_samples"]
        e_i = min(float(m.get("event_score", 0.0)), 1.0)
        l_val = max(0.0, float(m.get("val_loss", 0.1)))

        sample_factor = n_i / total_samples
        event_factor = np.exp(beta * e_i)
        quality_factor = 1.0 / (1.0 + l_val)

        raw_weights[i] = sample_factor * event_factor * quality_factor

    # Normalize weights so they sum to 1.0
    weight_sum = np.sum(raw_weights)
    if weight_sum <= 0:
        return np.ones(num_clients) / num_clients

    return raw_weights / weight_sum


def aggregate_befa(
    client_updates: List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]],
    beta: float = 0.3
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    """
    Aggregates participating client model states using BEFA weighting.

    Args:
        client_updates: List of tuples (client_state_dict, metadata_dict)
        beta: event importance scaling coefficient (default: 0.3)

    Returns:
        (aggregated_global_state, normalized_client_weights)
    """
    if not client_updates:
        raise ValueError("Cannot perform BEFA aggregation on empty client updates.")

    num_clients = len(client_updates)
    first_dict = client_updates[0][0]
    ref_shapes = {k: v.shape for k, v in first_dict.items()}

    # Compute normalized BEFA weights
    metadata_list = [meta for _, meta in client_updates]
    weights = compute_befa_weights(metadata_list, beta=beta)

    # Initialize accumulator
    aggregated_dict: Dict[str, torch.Tensor] = {}
    for key in ref_shapes.keys():
        aggregated_dict[key] = torch.zeros(ref_shapes[key], dtype=torch.float32)

    # Weighted accumulation: W_global = sum w_i * W_i
    for idx, (c_dict, _) in enumerate(client_updates):
        w_i = float(weights[idx])
        for key in ref_shapes.keys():
            param = c_dict[key].to(torch.float32)
            aggregated_dict[key].add_(param, alpha=w_i)

    # Sanity check for numerical stability
    for key, tensor in aggregated_dict.items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise RuntimeError(f"NaN/Inf generated in BEFA aggregation for key '{key}'")

    return aggregated_dict, weights
