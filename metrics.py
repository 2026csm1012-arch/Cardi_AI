"""
Evaluation metrics for Ejection Fraction (EF) video regression.
"""

import math
from typing import Dict

import numpy as np


def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute MAE, MSE, RMSE, R2, and Pearson correlation."""
    diff = preds - targets
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(math.sqrt(mse))

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

    if np.std(preds) > 1e-8 and np.std(targets) > 1e-8:
        pearson_r = float(np.corrcoef(preds, targets)[0, 1])
    else:
        pearson_r = 0.0

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
    }
