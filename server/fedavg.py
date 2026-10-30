"""
Sample-Count-Weighted Federated Averaging (FedAvg).
Implements standard McMahan et al. FedAvg baseline:
W_global = sum_{i=1}^K (n_i / sum_j n_j) * W_i
"""

from typing import List, Dict, Tuple, Optional
import torch


def validate_client_update(
    state_dict: Dict[str, torch.Tensor],
    reference_shapes: Dict[str, torch.Size]
) -> None:
    """
    Validates parameter names, shapes, and finiteness of a client model update.
    Raises ValueError if shapes mismatch or non-finite values (NaN / Inf) are detected.
    """
    if not state_dict:
        raise ValueError("Client state_dict is empty.")

    # Check key consistency
    if set(state_dict.keys()) != set(reference_shapes.keys()):
        missing = set(reference_shapes.keys()) - set(state_dict.keys())
        extra = set(state_dict.keys()) - set(reference_shapes.keys())
        raise ValueError(f"State dict keys mismatch. Missing: {missing}, Extra: {extra}")

    # Check shapes and finiteness
    for key, tensor in state_dict.items():
        expected_shape = reference_shapes[key]
        if tensor.shape != expected_shape:
            raise ValueError(f"Tensor shape mismatch for '{key}': expected {expected_shape}, got {tensor.shape}")

        if torch.isnan(tensor).any():
            raise ValueError(f"Non-finite update: NaN detected in '{key}'")
        if torch.isinf(tensor).any():
            raise ValueError(f"Non-finite update: Inf detected in '{key}'")


def aggregate_fedavg(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]]
) -> Dict[str, torch.Tensor]:
    """
    Performs sample-count-weighted Federated Averaging across participating clients.

    Args:
        client_updates: List of tuples (client_state_dict, num_samples_n_i)

    Returns:
        aggregated_state_dict: Global model parameters
    """
    if not client_updates:
        raise ValueError("Cannot perform FedAvg on an empty list of client updates.")

    num_clients = len(client_updates)
    total_samples = sum(n_i for _, n_i in client_updates)
    if total_samples <= 0:
        raise ValueError(f"Total training samples must be > 0. Got: {total_samples}")

    # Use first client as reference for keys and shapes
    first_dict = client_updates[0][0]
    ref_shapes = {k: v.shape for k, v in first_dict.items()}

    # Validate all client updates
    for client_idx, (c_dict, n_i) in enumerate(client_updates):
        if n_i <= 0:
            raise ValueError(f"Client {client_idx} reported invalid sample count: {n_i}")
        validate_client_update(c_dict, ref_shapes)

    # Initialize accumulator dictionary on CPU float32
    aggregated_dict: Dict[str, torch.Tensor] = {}
    for key in ref_shapes.keys():
        aggregated_dict[key] = torch.zeros(ref_shapes[key], dtype=torch.float32)

    # Accumulate weighted parameters: W_global = sum (n_i / total_n) * W_i
    for c_dict, n_i in client_updates:
        weight = float(n_i) / float(total_samples)
        for key in ref_shapes.keys():
            param = c_dict[key].to(torch.float32)
            aggregated_dict[key].add_(param, alpha=weight)

    # Final sanity check on aggregated weights
    for key, tensor in aggregated_dict.items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise RuntimeError(f"Numerical error: NaN/Inf produced in aggregated global parameter '{key}'")

    return aggregated_dict
