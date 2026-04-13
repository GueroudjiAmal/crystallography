"""
Classification and operational metrics for hit finder evaluation.
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> dict:
    """Compute full classification metrics.

    Args:
        y_true: Ground-truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_score: Continuous scores (probability of class 1) for AUC.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, pr_auc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        prec, rec, pr_thresholds = precision_recall_curve(y_true, y_score)
        metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds}
        metrics["pr_curve"] = {"precision": prec, "recall": rec, "thresholds": pr_thresholds}

    return metrics


def measure_inference_time(
    predict_fn,
    images: list,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> dict:
    """Measure per-image inference time.

    Args:
        predict_fn: Callable that takes a single image and returns a prediction.
        images: List of images to sample from.
        n_warmup: Number of warmup iterations (not timed).
        n_runs: Number of timed iterations.

    Returns:
        Dict with mean_ms, std_ms, throughput_per_sec.
    """
    n_runs = min(n_runs, len(images))
    indices = np.random.choice(len(images), n_warmup + n_runs, replace=True)

    # Warmup
    for i in range(n_warmup):
        predict_fn(images[indices[i]])

    # Timed runs
    times = []
    for i in range(n_warmup, n_warmup + n_runs):
        t0 = time.perf_counter()
        predict_fn(images[indices[i]])
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    return {
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "throughput_per_sec": 1000.0 / float(times_ms.mean()),
    }
