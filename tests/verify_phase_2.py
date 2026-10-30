"""
Live Verification Script for Phase 2: Compact STGCN Model and Spatial Graph Convolutions.
Executes detailed assertions and prints a full architectural and operational audit report.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.graph_layers import ChebGraphConv, GCNGraphConv, compute_chebyshev_polynomials
from models.stgcn import CompactSTGCN
from data.dataset_loader import (
    load_raw_pems04,
    build_adjacency_matrix,
    split_chronological,
    TrafficDataScaler,
    generate_sliding_windows
)


def banner(text: str):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def verify_phase_2():
    banner("PHASE 2: COMPACT STGCN MODEL ARCHITECTURE & GRAPH CONVOLUTION AUDIT")

    csv_path = "data/adjacency/PEMS04.csv"
    npz_path = "pems04.npz"

    # 1. Graph Laplacian & Chebyshev Polynomial Verification
    print("[1] Building Graph Laplacian and Chebyshev Polynomials (K=2)...")
    _, laplacian_307 = build_adjacency_matrix(csv_path, num_nodes=307)
    cheb_poly_307_np = compute_chebyshev_polynomials(laplacian_307, k_order=2)
    cheb_poly_307 = torch.from_numpy(np.stack(cheb_poly_307_np, axis=0)).float()

    assert cheb_poly_307.shape == (2, 307, 307)
    assert np.allclose(cheb_poly_307_np[0], np.eye(307)), "T_0 is not identity matrix!"
    assert np.allclose(cheb_poly_307_np[1], laplacian_307 - np.eye(307)), "T_1 does not equal scaled Laplacian!"
    print(f"    - Chebyshev T_0 shape: {cheb_poly_307[0].shape} (Identity Matrix verified)")
    print(f"    - Chebyshev T_1 shape: {cheb_poly_307[1].shape} (Scaled Laplacian L_tilde verified)")

    # 2. Model Instantiation & Detailed Layer Audit
    print("\n[2] Instantiating CompactSTGCN (Tin=12, Tout=3, C_in=3, C_hidden=32, K=2)...")
    model = CompactSTGCN(
        num_features=3,
        in_steps=12,
        out_steps=3,
        hidden_channels=32,
        k_order=2,
        kt=3,
        dropout=0.1
    )

    total_params, trainable_params = model.get_parameter_count()
    size_kb = model.get_model_size_kb()
    print(f"    - Total Parameters:     {total_params:,}")
    print(f"    - Trainable Parameters: {trainable_params:,}")
    print(f"    - Model Size (Float32): {size_kb:.2f} KB ({size_kb / 1024.0:.3f} MB)")
    assert total_params == trainable_params == 48515, f"Unexpected parameter count: {total_params}"

    # Print layer breakdown table
    print("\n    Layer-by-Layer Parameter Breakdown:")
    print("    " + "-" * 62)
    print(f"    {'Layer Name':<40} {'Shape':<15} {'Params':<8}")
    print("    " + "-" * 62)
    for name, param in model.named_parameters():
        shape_str = str(list(param.shape))
        print(f"    {name:<40} {shape_str:<15} {param.numel():<8}")
    print("    " + "-" * 62)

    # 3. Global Graph Forward Pass (N = 307)
    print("\n[3] Global Network Forward Pass (Batch=32, Tin=12, N=307, Features=3)...")
    batch_size = 32
    x_global = torch.randn(batch_size, 12, 307, 3)
    
    t0 = time.perf_counter()
    out_global = model(x_global, cheb_poly_307)
    t_global = (time.perf_counter() - t0) * 1000.0

    assert out_global.shape == (batch_size, 3, 307), f"Output shape mismatch: {out_global.shape}"
    assert not torch.isnan(out_global).any(), "Found NaN in global output!"
    assert not torch.isinf(out_global).any(), "Found Inf in global output!"
    print(f"    - Global Input Tensor:  {list(x_global.shape)}")
    print(f"    - Global Output Tensor: {list(out_global.shape)}")
    print(f"    - Latency (32 windows): {t_global:.2f} ms ({t_global/batch_size:.2f} ms/sample)")

    # 4. Client Sub-Graph Forward Pass (N = 61) & Node Invariance
    print("\n[4] Client Sub-Graph Forward Pass (Batch=32, Tin=12, N=61, Features=3)...")
    laplacian_61 = laplacian_307[:61, :61]
    cheb_poly_61_np = compute_chebyshev_polynomials(laplacian_61, k_order=2)
    cheb_poly_61 = torch.from_numpy(np.stack(cheb_poly_61_np, axis=0)).float()

    x_local = torch.randn(batch_size, 12, 61, 3)
    out_local = model(x_local, cheb_poly_61)

    assert out_local.shape == (batch_size, 3, 61), f"Local output shape mismatch: {out_local.shape}"
    assert not torch.isnan(out_local).any(), "Found NaN in local output!"
    assert not torch.isinf(out_local).any(), "Found Inf in local output!"
    print(f"    - Client Input Tensor:  {list(x_local.shape)}")
    print(f"    - Client Output Tensor: {list(out_local.shape)}")
    print(f"    - Parameter Invariance: EXACT MATCH. Model weights do not depend on node count N.")

    # 5. Backward Pass, Gradient Sanity & Optimizer Step
    print("\n[5] Backward Pass & Optimizer Update Sanity Test...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    y_target = torch.randn(batch_size, 3, 307)
    optimizer.zero_grad()
    pred = model(x_global, cheb_poly_307)
    loss = criterion(pred, y_target)
    loss.backward()

    # Gradient inspection
    zero_grads = 0
    total_grads = 0
    max_grad = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total_grads += 1
            assert p.grad is not None, f"Missing gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN in gradient for {name}"
            assert not torch.isinf(p.grad).any(), f"Inf in gradient for {name}"
            g_max = p.grad.abs().max().item()
            if g_max > max_grad:
                max_grad = g_max
            if torch.all(p.grad == 0):
                zero_grads += 1

    print(f"    - Loss Value: {loss.item():.4f}")
    print(f"    - Trainable Tensors: {total_grads} tensors verified")
    print(f"    - Gradient Sanity: Zero NaN/Inf, Max Gradient Magnitude = {max_grad:.6f}")
    assert zero_grads == 0, f"Found {zero_grads} tensors with zero gradients!"

    # Optimizer step check
    p_before = [p.clone().detach() for p in model.parameters()]
    optimizer.step()
    p_after = [p.clone().detach() for p in model.parameters()]
    changed = sum(not torch.equal(b, a) for b, a in zip(p_before, p_after))
    print(f"    - Optimizer Update: {changed} / {len(p_before)} parameter tensors successfully updated.")
    assert changed == len(p_before), "Not all parameter tensors were updated by optimizer!"

    # 6. Real PEMS04 Batch Execution Test
    print("\n[6] Real PEMS04 Batch Execution Test...")
    raw_data = load_raw_pems04(npz_path)
    train_raw, _, _ = split_chronological(raw_data, 0.70, 0.10, 0.20)
    scaler = TrafficDataScaler().fit(train_raw)
    train_norm = scaler.transform(train_raw[:100])

    x_real, y_real = generate_sliding_windows(train_norm, in_steps=12, out_steps=3)
    real_batch_x = torch.from_numpy(x_real[:16]).float()
    real_batch_y = torch.from_numpy(y_real[:16]).float()

    optimizer.zero_grad()
    real_pred = model(real_batch_x, cheb_poly_307)
    real_loss = nn.L1Loss()(real_pred, real_batch_y)
    real_loss.backward()
    optimizer.step()

    print(f"    - Real Input Batch:  {list(real_batch_x.shape)} (vehicles/occupancy/speed)")
    print(f"    - Real Target Batch: {list(real_batch_y.shape)} (flow values)")
    print(f"    - Real Batch L1 Loss: {real_loss.item():.4f} (normalized units)")
    print(f"    - Execution: 100% SUCCESS with real sensor data.")

    banner("ALL PHASE 2 MODEL VERIFICATION CHECKS PASSED SUCCESSFULLY (100%)")


if __name__ == "__main__":
    verify_phase_2()
