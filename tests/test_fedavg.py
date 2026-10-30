"""
Unit tests for Phase 4: Standard Federated Averaging (FedAvg) and Communication Accounting.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.fedavg import aggregate_fedavg, validate_client_update
from communication.serialization import (
    serialize_model_state,
    deserialize_model_state,
    measure_payload_bytes
)
from models.stgcn import CompactSTGCN


class TestFedAvg(unittest.TestCase):

    def test_01_two_client_known_numerical_aggregation(self):
        """
        Verify two-client sample-weighted FedAvg on known numerical tensors.
        Client 1: n1 = 100, W1 = [[2.0, 4.0]]
        Client 2: n2 = 300, W2 = [[6.0, 8.0]]
        Expected: W_global = (100/400)*W1 + (300/400)*W2 = 0.25*W1 + 0.75*W2
        = 0.25*[2, 4] + 0.75*[6, 8] = [0.5 + 4.5, 1.0 + 6.0] = [5.0, 7.0]
        """
        c1 = {"weight": torch.tensor([[2.0, 4.0]])}
        c2 = {"weight": torch.tensor([[6.0, 8.0]])}

        updates = [(c1, 100), (c2, 300)]
        global_dict = aggregate_fedavg(updates)

        expected = torch.tensor([[5.0, 7.0]])
        self.assertTrue(torch.allclose(global_dict["weight"], expected, atol=1e-5))

    def test_02_five_client_unequal_samples_aggregation(self):
        """
        Verify five-client aggregation with unequal sample counts.
        """
        sample_counts = [50, 150, 200, 100, 500]  # total = 1000
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        updates = []
        expected_val = 0.0
        for val, count in zip(values, sample_counts):
            c_dict = {"param": torch.full((3, 3), val)}
            updates.append((c_dict, count))
            expected_val += (count / 1000.0) * val

        # Expected: 0.05*1 + 0.15*2 + 0.20*3 + 0.10*4 + 0.50*5 = 0.05 + 0.30 + 0.60 + 0.40 + 2.50 = 3.85
        global_dict = aggregate_fedavg(updates)
        expected_tensor = torch.full((3, 3), expected_val)

        self.assertTrue(torch.allclose(global_dict["param"], expected_tensor, atol=1e-5))

    def test_03_identical_client_updates_invariance(self):
        """
        If all clients send identical weights, the aggregated model must be identically equal.
        """
        base_tensor = torch.randn(4, 8)
        updates = [
            ({"layer.weight": base_tensor.clone()}, 120),
            ({"layer.weight": base_tensor.clone()}, 250),
            ({"layer.weight": base_tensor.clone()}, 80)
        ]
        global_dict = aggregate_fedavg(updates)
        self.assertTrue(torch.allclose(global_dict["layer.weight"], base_tensor, atol=1e-6))

    def test_04_invalid_tensor_dimension_rejection(self):
        """
        Aggregation must reject client updates with mismatched tensor shapes.
        """
        c1 = {"weight": torch.randn(4, 4)}
        c2 = {"weight": torch.randn(4, 5)}  # Invalid shape

        updates = [(c1, 100), (c2, 100)]
        with self.assertRaises(ValueError):
            aggregate_fedavg(updates)

    def test_05_non_finite_tensor_rejection(self):
        """
        Aggregation must reject client updates containing NaN or Inf.
        """
        # Test NaN
        c_nan = {"weight": torch.tensor([[1.0, float('nan')]])}
        c_valid = {"weight": torch.tensor([[1.0, 2.0]])}
        with self.assertRaises(ValueError):
            aggregate_fedavg([(c_nan, 100), (c_valid, 100)])

        # Test Inf
        c_inf = {"weight": torch.tensor([[1.0, float('inf')]])}
        with self.assertRaises(ValueError):
            aggregate_fedavg([(c_inf, 100), (c_valid, 100)])

    def test_06_serialization_and_byte_measurement(self):
        """
        Verify exact payload serialization, deserialization integrity, and byte accounting.
        """
        model = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=32)
        state_dict = model.state_dict()

        # Measure serialized payload size
        payload = serialize_model_state(state_dict)
        size_bytes = len(payload)
        measured_bytes = measure_payload_bytes(state_dict)

        self.assertEqual(size_bytes, measured_bytes)
        self.assertGreater(size_bytes, 100_000, "CompactSTGCN state_dict should be > 100 KB")
        self.assertLess(size_bytes, 400_000, "CompactSTGCN state_dict should be < 400 KB")

        # Verify deserialization restores identical tensors
        restored = deserialize_model_state(payload)
        for k in state_dict.keys():
            self.assertTrue(torch.equal(state_dict[k], restored[k]), f"Tensor mismatch for key '{k}' after serialization")

    def test_07_global_model_redistribution_and_loading(self):
        """
        Verify that aggregated global parameters can be directly loaded into fresh model instances.
        """
        model_a = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=16)
        model_b = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=16)

        updates = [
            (model_a.state_dict(), 500),
            (model_b.state_dict(), 500)
        ]
        global_state = aggregate_fedavg(updates)

        global_model = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=16)
        global_model.load_state_dict(global_state)

        # Check all parameters match aggregated tensors
        for name, param in global_model.named_parameters():
            self.assertTrue(torch.allclose(param, global_state[name]))


if __name__ == '__main__':
    unittest.main()
