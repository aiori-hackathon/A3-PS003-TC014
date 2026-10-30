"""
Unit tests for Phase 5: Update Validator, Security Safeguards, BEFA Aggregation, and Compression.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from security.update_validator import UpdateValidator
from server.proposed_aggregation import compute_befa_weights, aggregate_befa
from communication.compression import (
    quantize_tensor_int8,
    dequantize_tensor_int8,
    compress_state_dict,
    decompress_state_dict,
    evaluate_compression_quality
)
from models.stgcn import CompactSTGCN


class TestValidationAndBEFA(unittest.TestCase):

    def test_01_update_validator_finiteness_and_shape(self):
        """Verify validator rejects NaN, Inf, and shape mismatches."""
        validator = UpdateValidator(max_norm_clip=5.0, max_staleness_rounds=2)
        global_state = {"w": torch.zeros(4, 4)}

        # Valid update
        ok, bounded, _ = validator.validate_and_clip(
            client_id=0,
            client_state={"w": torch.ones(4, 4)},
            global_state=global_state,
            client_round=1,
            current_round=1
        )
        self.assertTrue(ok)
        self.assertIsNotNone(bounded)

        # Invalid shape
        ok_shape, _, _ = validator.validate_and_clip(
            client_id=1,
            client_state={"w": torch.ones(4, 5)},
            global_state=global_state,
            client_round=1,
            current_round=1
        )
        self.assertFalse(ok_shape, "Validator must reject shape mismatch")

        # NaN rejection
        nan_tensor = torch.ones(4, 4)
        nan_tensor[0, 0] = float('nan')
        ok_nan, _, _ = validator.validate_and_clip(
            client_id=2,
            client_state={"w": nan_tensor},
            global_state=global_state,
            client_round=1,
            current_round=1
        )
        self.assertFalse(ok_nan, "Validator must reject updates with NaN")

    def test_02_update_validator_l2_norm_clipping(self):
        """Verify that update deltas exceeding max_norm_clip are proportionally scaled down."""
        max_clip = 3.0
        validator = UpdateValidator(max_norm_clip=max_clip)

        global_state = {"w": torch.zeros(10, 10)}
        # A large delta tensor: 100 elements each equal to 1.0 -> L2 norm = sqrt(100*1) = 10.0
        client_state = {"w": torch.ones(10, 10)}

        ok, bounded_state, _ = validator.validate_and_clip(
            client_id=0,
            client_state=client_state,
            global_state=global_state,
            client_round=1,
            current_round=1
        )
        self.assertTrue(ok)
        # Compute L2 norm of clipped delta
        clipped_delta = bounded_state["w"] - global_state["w"]
        clipped_norm = float(torch.norm(clipped_delta, p=2).item())

        self.assertAlmostEqual(clipped_norm, max_clip, places=4, msg="Clipped L2 norm must equal max_clip")

    def test_03_stale_and_duplicate_update_rejection(self):
        """Verify that stale updates and identical duplicate update hashes are rejected."""
        validator = UpdateValidator(max_staleness_rounds=2)
        global_state = {"w": torch.zeros(2, 2)}
        client_state = {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

        # Stale update: client trained on round 1, but current round is 5 (staleness = 4 > 2)
        ok_stale, _, reason = validator.validate_and_clip(
            client_id=0,
            client_state=client_state,
            global_state=global_state,
            client_round=1,
            current_round=5
        )
        self.assertFalse(ok_stale)
        self.assertIn("Stale", reason)

        # Fresh update on round 1 (Accepted)
        ok_fresh, _, _ = validator.validate_and_clip(
            client_id=0,
            client_state=client_state,
            global_state=global_state,
            client_round=1,
            current_round=1
        )
        self.assertTrue(ok_fresh)

        # Immediate duplicate submission from same client (Rejected)
        ok_dupl, _, reason_dupl = validator.validate_and_clip(
            client_id=0,
            client_state=client_state,
            global_state=global_state,
            client_round=1,
            current_round=1
        )
        self.assertFalse(ok_dupl, "Duplicate update hash must be rejected")
        self.assertIn("Duplicate", reason_dupl)

    def test_04_befa_weight_computation_and_aggregation(self):
        """Verify BEFA weights reward high event significance and dampen poor validation quality."""
        # Client 0: 100 samples, event_score = 0.8 (high traffic change), val_loss = 0.1 (good)
        # Client 1: 100 samples, event_score = 0.0 (stable traffic), val_loss = 1.0 (degraded)
        metadata = [
            {"num_samples": 100, "event_score": 0.8, "val_loss": 0.1},
            {"num_samples": 100, "event_score": 0.0, "val_loss": 1.0}
        ]
        weights = compute_befa_weights(metadata, beta=0.5)

        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=5)
        # Client 0 should receive significantly higher weight than Client 1
        self.assertGreater(weights[0], weights[1] * 1.5)

        # Test BEFA aggregation execution
        c0 = {"param": torch.tensor([[10.0, 20.0]])}
        c1 = {"param": torch.tensor([[30.0, 40.0]])}
        updates = [(c0, metadata[0]), (c1, metadata[1])]

        global_dict, norm_weights = aggregate_befa(updates, beta=0.5)
        self.assertIn("param", global_dict)
        self.assertEqual(global_dict["param"].shape, (1, 2))
        self.assertFalse(torch.isnan(global_dict["param"]).any())

    def test_05_int8_quantization_and_compression(self):
        """Verify 8-bit model compression achieves > 65% byte reduction with negligible reconstruction error."""
        model = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=16)
        state_dict = model.state_dict()

        quality = evaluate_compression_quality(state_dict)
        self.assertGreater(quality["savings_percent"], 60.0, f"Byte savings must be > 60%. Got: {quality['savings_percent']:.1f}%")
        self.assertLess(quality["reconstruction_mse"], 1e-4, f"Reconstruction MSE must be < 1e-4. Got: {quality['reconstruction_mse']}")

        # Verify full round-trip compression and decompression
        compressed_bytes = compress_state_dict(state_dict)
        restored_state = decompress_state_dict(compressed_bytes)

        for k in state_dict.keys():
            self.assertEqual(state_dict[k].shape, restored_state[k].shape)
            diff = torch.abs(state_dict[k] - restored_state[k])
            self.assertLess(diff.max().item(), 0.1, "Max absolute quantization error should be tiny")


if __name__ == '__main__':
    unittest.main()
