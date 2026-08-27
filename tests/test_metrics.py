import numpy as np
import pytest

from gtpro.metrics import (
    binary_classification_metrics,
    multilabel_classification_metrics,
    regression_metrics,
)


def test_binary_metrics_known_perfect_predictions():
    metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["warnings"] == []


def test_binary_single_class_is_explicitly_unavailable():
    metrics = binary_classification_metrics([1, 1], [0.6, 0.7])
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert "one class" in metrics["warnings"][0]


def test_multilabel_masks_missing_values_and_reports_valid_tasks():
    truth = np.asarray([[0, np.nan], [1, 1], [0, np.nan], [1, 1]])
    scores = np.asarray([[0.1, 0.2], [0.9, 0.8], [0.2, 0.3], [0.8, 0.7]])
    metrics = multilabel_classification_metrics(truth, scores, ["valid", "single"])
    assert metrics["valid_tasks"] == 1
    assert metrics["macro_roc_auc"] == pytest.approx(1.0)
    assert metrics["tasks"]["valid"]["valid_labels"] == 4
    assert metrics["tasks"]["single"]["roc_auc"] is None


def test_regression_metrics_known_values_and_constant_r2_warning():
    metrics = regression_metrics([1, 2, 3], [1, 2, 4])
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["r2"] == pytest.approx(0.5)

    constant = regression_metrics([2, 2], [2, 3])
    assert constant["r2"] is None
    assert constant["warnings"]


def test_metric_shape_mismatch_fails():
    with pytest.raises(ValueError, match="shapes differ"):
        regression_metrics([1, 2], [1])
