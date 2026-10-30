"""
Lightweight 8-Bit Model Update Compression Module.
Implements uniform int8 quantization to reduce client upload payload bytes by ~70%.
"""

import io
from typing import Dict, Tuple, Any
import numpy as np
import torch


def quantize_tensor_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """
    Uniformly quantizes a float32 tensor to uint8 with per-tensor min/max scales.
    Returns: (quantized_uint8_tensor, min_val, max_val)
    """
    min_val = float(tensor.min().item())
    max_val = float(tensor.max().item())
    span = max_val - min_val

    if span < 1e-7:
        # Uniform tensor
        q = torch.zeros_like(tensor, dtype=torch.uint8)
        return q, min_val, max_val

    # Normalize to [0, 255] and cast to uint8
    norm = (tensor - min_val) / span
    q = torch.clamp((norm * 255.0).round(), 0, 255).to(torch.uint8)
    return q, min_val, max_val


def dequantize_tensor_int8(
    q: torch.Tensor,
    min_val: float,
    max_val: float,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Dequantizes uint8 tensor back to float32 using min/max scales.
    """
    span = max_val - min_val
    if span < 1e-7:
        return torch.full(q.shape, min_val, dtype=dtype, device=q.device)

    norm = q.to(dtype) / 255.0
    return min_val + (norm * span)


def compress_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
    """
    Quantizes all floating-point tensors in state_dict and serializes to byte stream.
    """
    compressed_dict = {}
    for key, tensor in state_dict.items():
        if tensor.is_floating_point():
            q, min_v, max_v = quantize_tensor_int8(tensor.cpu())
            compressed_dict[key] = {
                "q": q,
                "min": min_v,
                "max": max_v,
                "is_quantized": True
            }
        else:
            compressed_dict[key] = {
                "tensor": tensor.cpu(),
                "is_quantized": False
            }

    buffer = io.BytesIO()
    torch.save(compressed_dict, buffer)
    return buffer.getvalue()


def decompress_state_dict(payload: bytes, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Deserializes and dequantizes byte payload back into float32 state_dict.
    """
    buffer = io.BytesIO(payload)
    compressed_dict = torch.load(buffer, map_location=map_location, weights_only=False)

    restored = {}
    for key, entry in compressed_dict.items():
        if entry["is_quantized"]:
            restored[key] = dequantize_tensor_int8(
                entry["q"],
                entry["min"],
                entry["max"]
            )
        else:
            restored[key] = entry["tensor"]

    return restored


def evaluate_compression_quality(state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Measures compressed payload size, byte savings ratio, and reconstruction MSE.
    """
    import io
    # Raw uncompressed size
    raw_buf = io.BytesIO()
    torch.save(state_dict, raw_buf)
    raw_bytes = len(raw_buf.getvalue())

    # Compressed size
    compressed_payload = compress_state_dict(state_dict)
    comp_bytes = len(compressed_payload)

    # Decompress and compute reconstruction MSE
    restored = decompress_state_dict(compressed_payload)
    total_sq_err = 0.0
    total_elements = 0

    for k in state_dict.keys():
        diff = state_dict[k].cpu().float() - restored[k].cpu().float()
        total_sq_err += torch.sum(diff ** 2).item()
        total_elements += diff.numel()

    mse = total_sq_err / total_elements

    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": comp_bytes,
        "savings_percent": ((raw_bytes - comp_bytes) / raw_bytes) * 100.0,
        "compression_ratio": raw_bytes / comp_bytes,
        "reconstruction_mse": mse
    }
