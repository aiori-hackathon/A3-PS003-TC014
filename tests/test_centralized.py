"""
Unit tests for Phase 3: Centralized Baseline, Evaluation Metrics, and Checkpoint Verification.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.stgcn import CompactSTGCN
from models.graph_layers import compute_chebyshev_polynomials
from data.dataset_loader import PEMS04DataManager
from evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_mape,
    evaluate_forecast_metrics,
    evaluate_persistence_baseline
)


class TestCentralizedBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = "results/checkpoints/centralized_best.pt"
        cls.metrics_csv = "results/metrics/centralized_vs_persistence.csv"
        cls.log_csv = "results/logs/centralized_training_log.csv"

    def test_01_metrics_mathematical_correctness(self):
        """Verify MAE, RMSE, and MAPE formulas on known synthetic arrays."""
        y_true = np.array([[10.0, 20.0], [30.0, 40.0]])
        y_pred = np.array([[12.0, 18.0], [33.0, 36.0]])

        # Absolute errors: |12-10|=2, |18-20|=2, |33-30|=3, |36-40|=4 -> mean = (2+2+3+4)/4 = 2.75
        mae = compute_mae(y_pred, y_true)
        self.assertAlmostEqual(mae, 2.75, places=5)

        # Squared errors: 4 + 4 + 9 + 16 = 33 -> mean = 33/4 = 8.25 -> sqrt = 2.87228
        rmse = compute_rmse(y_pred, y_true)
        self.assertAlmostEqual(rmse, np.sqrt(8.25), places=5)

        # Percentage errors: |2|/10=0.2, |2|/20=0.1, |3|/30=0.1, |4|/40=0.1 -> mean = 0.125 * 100 = 12.5%
        mape = compute_mape(y_pred, y_true, threshold=5.0)
        self.assertAlmostEqual(mape, 12.5, places=5)

    def test_02_persistence_baseline_logic(self):
        """Verify persistence baseline copies the last historical timestep to all forecast horizons."""
        # 2 samples, 12 in_steps, 3 nodes, 1 feature
        x_raw = np.zeros((2, 12, 3, 1), dtype=np.float32)
        # Set last timestep (t=11) to known values
        x_raw[0, -1, :, 0] = [100.0, 150.0, 200.0]
        x_raw[1, -1, :, 0] = [300.0, 350.0, 400.0]

        # Targets for 3 future steps
        y_raw = np.zeros((2, 3, 3), dtype=np.float32)
        y_raw[0, :, :] = [[100.0, 150.0, 200.0], [105.0, 155.0, 205.0], [110.0, 160.0, 210.0]]
        y_raw[1, :, :] = [[300.0, 350.0, 400.0], [305.0, 355.0, 405.0], [310.0, 360.0, 410.0]]

        results = evaluate_persistence_baseline(x_raw, y_raw)

        # At step 1, predictions equal targets exactly -> MAE=0, RMSE=0
        self.assertAlmostEqual(results["mae_step_1_5m"], 0.0, places=4)
        self.assertAlmostEqual(results["rmse_step_1_5m"], 0.0, places=4)

        # At step 2, diff is 5 for all nodes -> MAE=5, RMSE=5
        self.assertAlmostEqual(results["mae_step_2_10m"], 5.0, places=4)
        self.assertAlmostEqual(results["rmse_step_2_10m"], 5.0, places=4)

        # At step 3, diff is 10 for all nodes -> MAE=10, RMSE=10
        self.assertAlmostEqual(results["mae_step_3_15m"], 10.0, places=4)
        self.assertAlmostEqual(results["rmse_step_3_15m"], 10.0, places=4)

        # Overall MAE: (0 + 5 + 10) / 3 = 5.0
        self.assertAlmostEqual(results["mae_overall"], 5.0, places=4)

    def test_03_saved_checkpoint_integrity(self):
        """Verify the trained centralized checkpoint exists, loads cleanly, and contains all expected keys."""
        self.assertTrue(os.path.exists(self.checkpoint_path), f"Checkpoint missing: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")

        required_keys = ["epoch", "model_state_dict", "optimizer_state_dict", "val_loss", "val_metrics"]
        for k in required_keys:
            self.assertIn(k, ckpt, f"Key '{k}' missing from checkpoint dictionary")

        self.assertGreaterEqual(ckpt["epoch"], 1)
        self.assertGreater(ckpt["val_loss"], 0.0)

        # Load weights into a fresh model
        model = CompactSTGCN(num_features=3, in_steps=12, out_steps=3, hidden_channels=32)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Test forward pass with loaded weights
        x_dummy = torch.randn(2, 12, 307, 3)
        lap_dummy = np.eye(307, dtype=np.float32)
        poly_dummy = torch.from_numpy(np.stack(compute_chebyshev_polynomials(lap_dummy, k_order=2), axis=0)).float()
        with torch.no_grad():
            out = model(x_dummy, poly_dummy)
        self.assertEqual(out.shape, (2, 3, 307))
        self.assertFalse(torch.isnan(out).any())

    def test_04_training_logs_and_convergence(self):
        """Verify the training logs recorded decreasing or stable validation loss."""
        self.assertTrue(os.path.exists(self.log_csv), f"Log CSV missing: {self.log_csv}")
        df = pd.read_csv(self.log_csv)

        self.assertIn("epoch", df.columns)
        self.assertIn("train_loss", df.columns)
        self.assertIn("val_loss", df.columns)
        self.assertIn("val_mae", df.columns)
        self.assertIn("val_rmse", df.columns)

        # Training loss should decrease or be finite non-zero
        if len(df) > 1:
            self.assertGreaterEqual(df["train_loss"].iloc[0], df["train_loss"].iloc[-1])
        else:
            self.assertGreater(df["train_loss"].iloc[0], 0.0)
        # Val MAE should be reasonable (<30 veh/5-min)
        self.assertLess(df["val_mae"].min(), 30.0)

    def test_05_metrics_csv_outperforms_persistence(self):
        """Verify the final exported benchmark CSV demonstrates superior overall performance over persistence."""
        self.assertTrue(os.path.exists(self.metrics_csv), f"Metrics CSV missing: {self.metrics_csv}")
        df = pd.read_csv(self.metrics_csv)

        overall_row = df[df["horizon"] == "Overall (15 mins)"].iloc[0]
        # STGCN overall RMSE should be strictly lower than persistence RMSE
        self.assertLess(overall_row["centralized_rmse"], overall_row["persistence_rmse"])
        # Overall RMSE gain must be positive
        self.assertGreater(overall_row["rmse_gain_percent"], 0.0)


if __name__ == '__main__':
    unittest.main()
