"""
Evaluation Metrics for Traffic Flow Forecasting.
Computes MAE, RMSE, and MAPE on the physical scale (vehicles / 5-min).
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch


def compute_mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean Absolute Error: (1/N) * sum(|y - y_hat|)
    """
    return float(np.mean(np.abs(preds - targets)))


def compute_rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Root Mean Square Error: sqrt((1/N) * sum((y - y_hat)^2))
    """
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def compute_mape(preds: np.ndarray, targets: np.ndarray, threshold: float = 5.0) -> float:
    """
    Mean Absolute Percentage Error (%): (1/N) * sum(|y - y_hat| / y) * 100
    Filters out observations with targets < threshold to prevent division by near-zero.
    """
    mask = targets > threshold
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100.0)


def evaluate_forecast_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 5.0
) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and MAPE overall and per forecasting horizon step.
    preds:   shape (num_samples, out_steps, num_nodes)
    targets: shape (num_samples, out_steps, num_nodes)
    """
    assert preds.shape == targets.shape, f"Shape mismatch: {preds.shape} vs {targets.shape}"
    out_steps = preds.shape[1]
    
    results = {
        "mae_overall": compute_mae(preds, targets),
        "rmse_overall": compute_rmse(preds, targets),
        "mape_overall": compute_mape(preds, targets, threshold)
    }
    
    for h in range(out_steps):
        p_h = preds[:, h, :]
        t_h = targets[:, h, :]
        step_min = (h + 1) * 5  # 5-min steps
        results[f"mae_step_{h+1}_{step_min}m"] = compute_mae(p_h, t_h)
        results[f"rmse_step_{h+1}_{step_min}m"] = compute_rmse(p_h, t_h)
        results[f"mape_step_{h+1}_{step_min}m"] = compute_mape(p_h, t_h, threshold)

    return results


def evaluate_persistence_baseline(
    x_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
    target_channel: int = 0
) -> Dict[str, float]:
    """
    Computes historical persistence baseline:
    Predictions for all future horizons are set to the last observed timestep in input window:
    y_hat_{t+h} = x_{t} for all h in [1 .. out_steps]
    
    x_test_raw: shape (num_samples, in_steps, num_nodes, num_features)
    y_test_raw: shape (num_samples, out_steps, num_nodes)
    """
    # Last observed value in window: x[:, -1, :, target_channel] -> shape (samples, nodes)
    last_val = x_test_raw[:, -1, :, target_channel]
    out_steps = y_test_raw.shape[1]
    
    # Broadcast across out_steps: shape (samples, out_steps, nodes)
    preds = np.repeat(last_val[:, np.newaxis, :], out_steps, axis=1)
    
    return evaluate_forecast_metrics(preds, y_test_raw)
