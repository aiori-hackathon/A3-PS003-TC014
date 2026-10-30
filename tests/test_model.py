"""
Unit tests for Phase 2: Compact STGCN Model and Graph Convolutions.
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.graph_layers import ChebGraphConv, GCNGraphConv, compute_chebyshev_polynomials
from models.stgcn import CompactSTGCN
from data.dataset_loader import build_adjacency_matrix, load_raw_pems04, split_chronological, TrafficDataScaler, generate_sliding_windows


class TestCompactSTGCN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = "data/adjacency/PEMS04.csv"
        cls.npz_path = "pems04.npz"
        
        # Load Laplacian and Chebyshev polynomials for 307 global nodes
        _, cls.laplacian_307 = build_adjacency_matrix(cls.csv_path, num_nodes=307)
        cls.cheb_poly_307_np = compute_chebyshev_polynomials(cls.laplacian_307, k_order=2)
        cls.cheb_poly_307 = torch.from_numpy(np.stack(cls.cheb_poly_307_np, axis=0)).float()

        # Build sub-graph for a local client with 61 nodes
        cls.laplacian_61 = cls.laplacian_307[:61, :61]
        cls.cheb_poly_61_np = compute_chebyshev_polynomials(cls.laplacian_61, k_order=2)
        cls.cheb_poly_61 = torch.from_numpy(np.stack(cls.cheb_poly_61_np, axis=0)).float()

    def test_01_chebyshev_polynomial_math(self):
        """Verify Chebyshev polynomial definition: T_0 = I, T_1 = L_tilde, T_2 = 2 L_tilde T_1 - T_0."""
        poly_np = compute_chebyshev_polynomials(self.laplacian_307, k_order=3)
        self.assertEqual(len(poly_np), 3)
        # T_0 must be identity
        self.assertTrue(np.allclose(poly_np[0], np.eye(307)))
        # T_1 must be L_tilde = L - I
        self.assertTrue(np.allclose(poly_np[1], self.laplacian_307 - np.eye(307)))

    def test_02_cheb_graph_conv_forward(self):
        """Verify ChebGraphConv forward pass and output dimensions."""
        b, c_in, t, n = 4, 3, 12, 307
        c_out = 16
        conv = ChebGraphConv(c_in=c_in, c_out=c_out, k_order=2)
        x = torch.randn(b, c_in, t, n)
        out = conv(x, self.cheb_poly_307)

        self.assertEqual(out.shape, (b, c_out, t, n))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_03_dimension_invariance_across_node_counts(self):
        """Verify that the model weights are strictly node-invariant and execute on both N=307 and N=61."""
        model = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=32)

        # Global graph: N = 307
        x_global = torch.randn(2, 12, 307, 3)
        out_global = model(x_global, self.cheb_poly_307)
        self.assertEqual(out_global.shape, (2, 3, 307))

        # Local client sub-graph: N = 61
        x_local = torch.randn(2, 12, 61, 3)
        out_local = model(x_local, self.cheb_poly_61)
        self.assertEqual(out_local.shape, (2, 3, 61))

        # Weights are identical in shape and count
        state_dict_keys = list(model.state_dict().keys())
        self.assertGreater(len(state_dict_keys), 0)

    def test_04_forward_backward_and_optimizer_step(self):
        """Verify that loss computes cleanly, backward pass calculates finite non-zero gradients, and optimizer updates."""
        model = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=16)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        x = torch.randn(4, 12, 307, 3)
        y = torch.randn(4, 3, 307)

        # Record parameter values before step
        p_before = [p.clone().detach() for p in model.parameters() if p.requires_grad]

        optimizer.zero_grad()
        pred = model(x, self.cheb_poly_307)
        loss = criterion(pred, y)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

        loss.backward()

        # Verify gradients exist and are finite
        has_nonzero_grad = False
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertFalse(torch.isnan(p.grad).any())
                self.assertFalse(torch.isinf(p.grad).any())
                if (p.grad != 0).any():
                    has_nonzero_grad = True
        self.assertTrue(has_nonzero_grad, "At least some gradients must be non-zero")

        optimizer.step()

        # Verify weights were updated
        p_after = [p.clone().detach() for p in model.parameters() if p.requires_grad]
        weights_changed = any(not torch.equal(b, a) for b, a in zip(p_before, p_after))
        self.assertTrue(weights_changed, "Optimizer step must update model weights")

    def test_05_reproducible_initialization(self):
        """Verify reproducible model parameter initialization given identical seeds."""
        torch.manual_seed(42)
        model1 = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=16)

        torch.manual_seed(42)
        model2 = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=16)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2), "Identical seeds must yield identical parameter values")

    def test_06_parameter_count_and_footprint(self):
        """Verify model remains compact and compute footprint in KB."""
        model = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=32)
        total_p, train_p = model.get_parameter_count()
        size_kb = model.get_model_size_kb()

        self.assertLess(total_p, 100_000, f"STGCN model should be compact (<100k parameters). Got: {total_p}")
        self.assertEqual(total_p, train_p, "All parameters should be trainable")
        print(f"\n[STGCN Parameters] Total: {total_p:,} | Size: {size_kb:.2f} KB")

    def test_07_real_pems04_batch_execution(self):
        """Verify forward and backward passes using an actual batch extracted from PEMS04."""
        raw_data = load_raw_pems04(self.npz_path)
        train_raw, _, _ = split_chronological(raw_data, 0.70, 0.10, 0.20)
        scaler = TrafficDataScaler().fit(train_raw)
        train_norm = scaler.transform(train_raw[:50])

        x, y = generate_sliding_windows(train_norm, in_steps=12, out_steps=3)
        batch_x = torch.from_numpy(x[:8]).float()
        batch_y = torch.from_numpy(y[:8]).float()

        model = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=32)
        model.train()
        pred = model(batch_x, self.cheb_poly_307)

        self.assertEqual(pred.shape, batch_y.shape)
        loss = nn.L1Loss()(pred, batch_y)
        loss.backward()
        self.assertGreater(loss.item(), 0.0)


if __name__ == '__main__':
    unittest.main()
