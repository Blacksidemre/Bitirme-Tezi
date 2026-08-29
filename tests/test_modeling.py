from __future__ import annotations

import numpy as np

from atakum_housing.modeling import regression_metrics


def test_regression_metrics_return_expected_values() -> None:
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 190.0, 310.0])

    metrics = regression_metrics(actual, predicted)

    assert metrics["mae"] == 10.0
    assert np.isclose(metrics["rmse"], 10.0)
    assert np.isclose(metrics["r2"], 0.985)
    assert np.isclose(metrics["mape"], (0.10 + 0.05 + 1 / 30) / 3)
