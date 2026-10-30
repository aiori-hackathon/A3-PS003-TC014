"""
Model State Serialization and Communication Byte Accounting.
Measures exact payload sizes in bytes for network transmission simulation.
"""

import io
from typing import Dict, Tuple
import torch


def serialize_model_state(state_dict: Dict[str, torch.Tensor]) -> bytes:
    """
    Serializes a PyTorch model state_dict into a byte stream using torch.save.
    Simulates the exact payload transmitted over the physical network.
    """
    buffer = io.BytesIO()
    # Move tensors to CPU before serializing to ensure cross-device consistency
    cpu_state = {k: v.cpu().detach() for k, v in state_dict.items()}
    torch.save(cpu_state, buffer)
    return buffer.getvalue()


def deserialize_model_state(payload: bytes, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Deserializes raw byte payload back into a PyTorch state_dict.
    """
    buffer = io.BytesIO(payload)
    return torch.load(buffer, map_location=map_location, weights_only=True)


def measure_payload_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    Measures the exact size in bytes of a serialized model state_dict.
    """
    return len(serialize_model_state(state_dict))
