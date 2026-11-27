import numpy as np


def metrics(target, prediction, complete_prediction=None):
    y, p = np.asarray(target, float), np.asarray(prediction, float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Metric inputs must be equal nonempty finite vectors.')
    error = p - y
    mse = float(np.mean(error**2))
    denominator = float(np.sum((y - y.mean())**2))
    nonzero = y != 0
    result = {'n': len(y), 'mae': float(np.abs(error).mean()), 'mse': mse, 'rmse': mse**.5,
              'r2': float(1 - np.sum(error**2) / denominator) if len(y) >= 2 and denominator > 0 else None,
              'mape': float(100 * np.mean(np.abs(error[nonzero] / y[nonzero]))) if nonzero.any() else None,
              'mape_zero_excluded': int((~nonzero).sum()),
              'pearson': float(np.corrcoef(y, p)[0,1]) if len(y) >= 2 and np.std(y) > 0 and np.std(p) > 0 else None,
              'bias': float(error.mean())}
    if complete_prediction is not None:
        complete = np.asarray(complete_prediction, float)
        if complete.shape != p.shape or not np.isfinite(complete).all():
            raise ValueError('Complete predictions must match each video in order.')
        complete_mae = float(np.mean(np.abs(complete - y)))
        delta = result['mae'] - complete_mae
        result.update(complete_mae=complete_mae, delta_mae=delta,
                      relative_delta_mae=delta / complete_mae if complete_mae > 0 else None,
                      prediction_shift=float(np.mean(np.abs(p - complete))))
    return result


def robustness(ratios, mae):
    r, values = np.asarray(ratios, float), np.asarray(mae, float)
    if len(r) < 2 or r.shape != values.shape or not np.isfinite(values).all():
        raise ValueError('At least two finite MAE points are required.')
    order = np.argsort(r)
    r, values = r[order], values[order]
    if not np.isfinite(r).all() or np.any(np.diff(r) <= 0) or r[0] != 0:
        raise ValueError('Unique increasing ratios starting at zero are required.')
    area = float(np.sum(np.diff(r) * (values[1:] + values[:-1]) / 2))
    return {'auc_mae': area, 'normalized_auc_mae': area / (r[-1] - r[0]),
            'auc_ratio_max': float(r[-1])}
