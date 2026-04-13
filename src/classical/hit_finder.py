"""
Hit finder wrapper around Peakfinder8.

Provides batch evaluation and threshold sweeping for ROC curve generation.
"""

import numpy as np
from src.classical.peakfinder8 import PeakFinder8


class PeakFinder8HitFinder:
    """Wraps PeakFinder8 for batch evaluation and threshold sweeping."""

    def __init__(self, **pf8_kwargs):
        self.pf8 = PeakFinder8(**pf8_kwargs)

    def count_peaks(self, image: np.ndarray) -> int:
        """Return the number of detected peaks for a single image."""
        result = self.pf8.find_peaks(image)
        return result.n_peaks

    def classify(self, image: np.ndarray, threshold: int | None = None) -> bool:
        """Classify a single image as hit or miss."""
        n_peaks = self.count_peaks(image)
        t = threshold if threshold is not None else self.pf8.n_peaks_threshold
        return n_peaks >= t

    def evaluate_batch(
        self,
        images: list[np.ndarray],
        labels: list[int],
        thresholds: list[int] | None = None,
    ) -> dict:
        """Evaluate on a batch of images.

        Args:
            images: List of 2D uint16 arrays.
            labels: Ground-truth labels (1=hit, 0=miss).
            thresholds: If provided, sweep these thresholds and return
                        per-threshold metrics for ROC curve.

        Returns:
            Dict with peak_counts and (optionally) per-threshold results.
        """
        peak_counts = []
        for img in images:
            peak_counts.append(self.count_peaks(img))

        peak_counts = np.array(peak_counts)
        labels = np.array(labels)

        if thresholds is None:
            thresholds = list(range(0, int(peak_counts.max()) + 2))

        results_per_threshold = []
        for t in thresholds:
            preds = (peak_counts >= t).astype(int)
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()

            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            precision = tp / max(tp + fp, 1)

            results_per_threshold.append({
                "threshold": t,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "tpr": float(tpr), "fpr": float(fpr),
                "precision": float(precision), "recall": float(tpr),
            })

        return {
            "peak_counts": peak_counts,
            "labels": labels,
            "thresholds": results_per_threshold,
        }
