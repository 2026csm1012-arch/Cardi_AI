import numpy as np
import pytest
from cardiac.evaluation.metrics import metrics, robustness


def test_known_metrics():
    m = metrics([10,20,30],[12,18,30],[10,20,30])
    assert m['mae'] == pytest.approx(4/3)
    assert m['mse'] == pytest.approx(8/3)
    assert m['rmse'] == pytest.approx((8/3)**.5)
    assert m['mape'] == pytest.approx(10)
    assert m['r2'] == pytest.approx(.96)
    assert m['bias'] == 0
    assert m['relative_delta_mae'] is None
    assert m['prediction_shift'] == pytest.approx(4/3)


def test_undefined_and_invalid():
    m = metrics([0,0],[2,2])
    assert m['mape'] is None and m['r2'] is None and m['pearson'] is None
    assert m['mape_zero_excluded'] == 2
    with pytest.raises(ValueError): metrics([],[])
    with pytest.raises(ValueError): metrics([1],[float('nan')])
    assert metrics([1],[1])['r2'] is None


def test_auc():
    m=robustness([.75,0,.25],[4,4,4])
    assert m['auc_mae'] == 3 and m['normalized_auc_mae'] == 4
    with pytest.raises(ValueError): robustness([0,0],[1,2])
