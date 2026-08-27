"""Auditable downstream metrics with explicit unavailable-value handling."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def _arrays(targets, predictions) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(targets, dtype=np.float64)
    scores = np.asarray(predictions, dtype=np.float64)
    if truth.shape != scores.shape:
        raise ValueError(f"target/prediction shapes differ: {truth.shape} vs {scores.shape}")
    return truth, scores


def binary_classification_metrics(targets, probabilities) -> dict[str, object]:
    truth, scores = _arrays(targets, probabilities)
    truth = truth.reshape(-1)
    scores = scores.reshape(-1)
    mask = np.isfinite(truth) & np.isfinite(scores)
    truth, scores = truth[mask], scores[mask]
    warnings: list[str] = []
    if truth.size == 0:
        warnings.append("no valid labels; ROC-AUC and PR-AUC are unavailable")
        roc_auc = pr_auc = None
    elif np.unique(truth).size < 2:
        warnings.append("only one class is present; ROC-AUC and PR-AUC are unavailable")
        roc_auc = pr_auc = None
    else:
        roc_auc = float(roc_auc_score(truth, scores))
        pr_auc = float(average_precision_score(truth, scores))
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "valid_labels": int(truth.size),
        "warnings": warnings,
    }


def multilabel_classification_metrics(
    targets,
    probabilities,
    target_names: Sequence[str] | None = None,
) -> dict[str, object]:
    truth, scores = _arrays(targets, probabilities)
    if truth.ndim != 2:
        raise ValueError("multilabel targets and predictions must be rank-2")
    names = list(target_names or (f"task_{index}" for index in range(truth.shape[1])))
    if len(names) != truth.shape[1]:
        raise ValueError("target_names length does not match multilabel width")
    task_results: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    for index, name in enumerate(names):
        result = binary_classification_metrics(truth[:, index], scores[:, index])
        task_results[name] = result
        warnings.extend(f"{name}: {message}" for message in result["warnings"])
    valid_roc = [result["roc_auc"] for result in task_results.values() if result["roc_auc"] is not None]
    valid_pr = [result["pr_auc"] for result in task_results.values() if result["pr_auc"] is not None]
    return {
        "macro_roc_auc": float(np.mean(valid_roc)) if valid_roc else None,
        "macro_pr_auc": float(np.mean(valid_pr)) if valid_pr else None,
        "valid_tasks": len(valid_roc),
        "total_tasks": truth.shape[1],
        "tasks": task_results,
        "warnings": warnings,
    }


def regression_metrics(targets, predictions) -> dict[str, object]:
    truth, values = _arrays(targets, predictions)
    truth = truth.reshape(-1)
    values = values.reshape(-1)
    mask = np.isfinite(truth) & np.isfinite(values)
    truth, values = truth[mask], values[mask]
    warnings: list[str] = []
    if truth.size == 0:
        warnings.append("no valid labels; regression metrics are unavailable")
        rmse = mae = r2 = None
    else:
        rmse = float(np.sqrt(mean_squared_error(truth, values)))
        mae = float(mean_absolute_error(truth, values))
        if truth.size < 2 or np.all(truth == truth[0]):
            warnings.append("R2 requires at least two non-constant labels; R2 is unavailable")
            r2 = None
        else:
            r2 = float(r2_score(truth, values))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "valid_labels": int(truth.size),
        "warnings": warnings,
    }


def compute_metrics(task_type: str, targets, predictions, target_names=None) -> dict[str, object]:
    if task_type == "binary_classification":
        return binary_classification_metrics(targets, predictions)
    if task_type == "multilabel_classification":
        return multilabel_classification_metrics(targets, predictions, target_names)
    if task_type == "regression":
        return regression_metrics(targets, predictions)
    raise ValueError(f"unsupported task type: {task_type}")
