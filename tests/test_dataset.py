"""
Unit tests for Phase 1: Dataset Validation, Adjacency Matrix, and Client Partitioning.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_loader import (
    load_raw_pems04,
    build_adjacency_matrix,
    split_chronological,
    TrafficDataScaler,
    generate_sliding_windows,
    PEMS04DataManager
)
from clients.partitioner import FederatedPartitioner


class TestPEMS04Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.npz_path = "pems04.npz"
        cls.csv_path = "data/adjacency/PEMS04.csv"

    def test_01_npz_loading_and_dimensions(self):
        """Verify NPZ loading, array shape, data type, and feature presence."""
        data = load_raw_pems04(self.npz_path)
        self.assertEqual(data.shape, (16992, 307, 3), "Shape must be exactly (16992, 307, 3)")
        self.assertEqual(data.dtype, np.float32, "Data type should be float32 for PyTorch compatibility")
        self.assertFalse(np.isnan(data).any(), "Raw data must not contain NaN values")
        self.assertFalse(np.isinf(data).any(), "Raw data must not contain infinite values")

    def test_02_feature_statistics_and_sanity(self):
        """Verify feature bounds and physical sanity."""
        data = load_raw_pems04(self.npz_path)
        flow = data[:, :, 0]
        occ = data[:, :, 1]
        speed = data[:, :, 2]

        self.assertGreaterEqual(flow.min(), 0.0, "Traffic flow must be non-negative")
        self.assertGreaterEqual(occ.min(), 0.0, "Occupancy must be non-negative")
        self.assertLessEqual(occ.max(), 1.0, "Occupancy must be <= 1.0 (proportion/percentage)")
        self.assertGreater(speed.min(), 0.0, "Speed readings must be strictly positive")

    def test_03_adjacency_matrix_validation(self):
        """Verify adjacency matrix construction, symmetry, self-loops, and graph Laplacian."""
        adj, laplacian = build_adjacency_matrix(self.csv_path, num_nodes=307)
        self.assertEqual(adj.shape, (307, 307), "Adjacency matrix must be 307x307")
        self.assertEqual(laplacian.shape, (307, 307), "Laplacian must be 307x307")
        
        # Symmetry check
        self.assertTrue(np.allclose(adj, adj.T, atol=1e-5), "Adjacency matrix must be symmetric")
        self.assertTrue(np.allclose(laplacian, laplacian.T, atol=1e-5), "Normalized Laplacian must be symmetric")

        # Diagonal check (self-loops)
        self.assertTrue(np.allclose(np.diag(adj), 1.0), "Adjacency diagonal must contain 1.0 self-loops")
        
        # Value ranges and finiteness
        self.assertTrue((adj >= 0.0).all() and (adj <= 1.0).all(), "Adjacency values must be in [0, 1]")
        self.assertFalse(np.isnan(adj).any(), "Adjacency matrix must not contain NaN")
        self.assertFalse(np.isinf(adj).any(), "Adjacency matrix must not contain Inf")
        self.assertFalse(np.isnan(laplacian).any(), "Laplacian must not contain NaN")

    def test_04_chronological_splitting(self):
        """Verify strict chronological splitting with 70% train, 10% val, 20% test."""
        data = load_raw_pems04(self.npz_path)
        train, val, test = split_chronological(data, 0.70, 0.10, 0.20)

        total_steps = len(data)
        self.assertEqual(len(train), int(total_steps * 0.70), "Train set must have 70% of timesteps")
        self.assertEqual(len(val), int(total_steps * 0.10), "Val set must have 10% of timesteps")
        self.assertEqual(len(train) + len(val) + len(test), total_steps, "All timesteps must be preserved")

        # Temporal integrity: train comes strictly before val, which comes strictly before test
        np.testing.assert_array_equal(train[-1], data[len(train) - 1])
        np.testing.assert_array_equal(val[0], data[len(train)])
        np.testing.assert_array_equal(test[0], data[len(train) + len(val)])

    def test_05_zero_data_leakage_scaling(self):
        """Verify scaler fits exclusively on training data and prevents leakage."""
        data = load_raw_pems04(self.npz_path)
        train, val, test = split_chronological(data, 0.70, 0.10, 0.20)

        scaler = TrafficDataScaler()
        scaler.fit(train)

        train_norm = scaler.transform(train)
        val_norm = scaler.transform(val)
        test_norm = scaler.transform(test)

        # Train normalized flow should have mean ~0 and std ~1
        self.assertTrue(np.allclose(train_norm[:, :, 0].mean(), 0.0, atol=1e-2), "Train normalized mean must be ~0")
        self.assertTrue(np.allclose(train_norm[:, :, 0].std(), 1.0, atol=1e-2), "Train normalized std must be ~1")

        # Test normalized mean will naturally reflect domain shift and must NOT be identically 0
        self.assertFalse(np.isclose(test_norm[:, :, 0].mean(), 0.0, atol=1e-3), "Test mean should not be identically 0 (proves no leak)")

        # Verify exact inversion
        inverted_flow = scaler.inverse_transform_flow(train_norm[:, :, 0])
        self.assertTrue(np.allclose(inverted_flow, train[:, :, 0], atol=1e-4), "Inverse transform must recover raw flow exactly")

    def test_06_sliding_window_dimensions(self):
        """Verify sliding window generator produces correct temporal input/output dimensions."""
        data = load_raw_pems04(self.npz_path)
        train, _, _ = split_chronological(data, 0.70, 0.10, 0.20)

        in_steps = 12   # 1 hour
        out_steps = 3   # 15 mins
        x, y = generate_sliding_windows(train[:100], in_steps=in_steps, out_steps=out_steps)

        # 100 timesteps - 15 + 1 = 86 samples
        self.assertEqual(x.shape, (86, in_steps, 307, 3), "X window shape mismatch")
        self.assertEqual(y.shape, (86, out_steps, 307), "Y target shape mismatch")

    def test_07_federated_iid_and_non_iid_partitioning(self):
        """Verify client partitioner creates valid, mutually exclusive, complete sensor partitions."""
        data = load_raw_pems04(self.npz_path)
        train, _, _ = split_chronological(data, 0.70, 0.10, 0.20)
        adj, lap = build_adjacency_matrix(self.csv_path, num_nodes=307)

        partitioner = FederatedPartitioner(num_clients=5, num_nodes=307, random_seed=42)
        
        # Test IID
        iid_parts = partitioner.partition_iid(train, adj, lap)
        self.assertEqual(len(iid_parts), 5, "Must create 5 virtual clients")
        
        all_sensors_iid = []
        for p in iid_parts:
            all_sensors_iid.extend(p.sensor_indices.tolist())
            self.assertEqual(p.local_adj.shape, (p.num_sensors, p.num_sensors))
        
        self.assertEqual(len(all_sensors_iid), 307, "Total sensor count must be 307")
        self.assertEqual(set(all_sensors_iid), set(range(307)), "All sensors 0..306 must be uniquely assigned")
        
        # Verify IID balance: mean flows across clients should be close
        flows_iid = [p.mean_flow for p in iid_parts]
        self.assertLess(max(flows_iid) - min(flows_iid), 10.0, "IID clients must have nearly identical mean flow (<10 veh/5min diff)")

        # Test Non-IID
        non_iid_parts = partitioner.partition_non_iid(train, adj, lap)
        self.assertEqual(len(non_iid_parts), 5)
        
        all_sensors_non_iid = []
        for p in non_iid_parts:
            all_sensors_non_iid.extend(p.sensor_indices.tolist())
        self.assertEqual(set(all_sensors_non_iid), set(range(307)))

        # Verify Non-IID heterogeneity: significant flow difference between lowest and highest tier
        flows_non_iid = [p.mean_flow for p in non_iid_parts]
        self.assertGreater(max(flows_non_iid) - min(flows_non_iid), 150.0, "Non-IID clients must exhibit significant volume tier divergence (>150 veh/5min diff)")


if __name__ == '__main__':
    unittest.main()
