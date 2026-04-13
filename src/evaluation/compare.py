"""
Side-by-side comparison of Peakfinder8 and ViT hit finders.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.classical.peakfinder8 import PeakFinder8
from src.evaluation.metrics import classification_metrics, measure_inference_time
from src.evaluation.visualize import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrices,
    plot_peak_count_histogram,
    print_comparison_table,
)


def evaluate_peakfinder8(
    images: list[np.ndarray],
    labels: np.ndarray,
    pf8_params: dict,
) -> dict:
    """Evaluate Peakfinder8 on a test set.

    Returns metrics dict with 'peak_counts' and classification metrics
    at the optimal threshold.
    """
    pf8 = PeakFinder8(**pf8_params)

    peak_counts = []
    print("Running Peakfinder8...")
    for img in tqdm(images, desc="PF8"):
        result = pf8.find_peaks(img)
        peak_counts.append(result.n_peaks)

    peak_counts = np.array(peak_counts)

    # Find optimal threshold by sweeping
    best_f1 = 0.0
    best_threshold = pf8_params.get("n_peaks_threshold", 15)

    max_peaks = int(peak_counts.max()) + 1
    for t in range(0, max_peaks + 1):
        preds = (peak_counts >= t).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # Use peak_counts as continuous scores for ROC/PR curves
    # Higher peak count = more likely a hit
    y_score = peak_counts.astype(float)
    # Normalize to [0,1] range for AUC computation
    if y_score.max() > y_score.min():
        y_score_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min())
    else:
        y_score_norm = y_score

    y_pred = (peak_counts >= best_threshold).astype(int)
    metrics = classification_metrics(labels, y_pred, y_score_norm)
    metrics["peak_counts"] = peak_counts
    metrics["optimal_threshold"] = best_threshold

    # Timing
    timing = measure_inference_time(
        lambda img: pf8.find_peaks(img),
        images,
        n_warmup=3,
        n_runs=min(50, len(images)),
    )
    metrics["timing"] = timing

    return metrics


def evaluate_vit(
    images_tensor: torch.Tensor,
    labels: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Evaluate ViT on a test set.

    Args:
        images_tensor: Preprocessed tensor of shape (N, 1, 224, 224).
        labels: Ground-truth labels.
        model: Trained ViT model.
        device: torch device.
        batch_size: Batch size for inference.

    Returns metrics dict with classification metrics.
    """
    model.eval()
    model = model.to(device)

    all_probs = []
    print("Running ViT inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(images_tensor), batch_size), desc="ViT"):
            batch = images_tensor[i : i + batch_size].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    y_score = np.array(all_probs)

    # Find optimal threshold
    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_score >= t).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    y_pred = (y_score >= best_t).astype(int)
    metrics = classification_metrics(labels, y_pred, y_score)
    metrics["optimal_threshold"] = best_t

    # Timing (single-image)
    def single_predict(img_tensor):
        with torch.no_grad():
            batch = img_tensor.unsqueeze(0).to(device)
            model(batch)

    timing = measure_inference_time(
        lambda _img: single_predict(images_tensor[0]),
        [None] * 50,  # dummy list, we always use the same image for timing
        n_warmup=5,
        n_runs=50,
    )
    metrics["timing"] = timing

    return metrics


def run_comparison(
    raw_images: list[np.ndarray],
    preprocessed_tensor: torch.Tensor,
    labels: np.ndarray,
    vit_model: torch.nn.Module,
    pf8_params: dict,
    device: torch.device,
    figures_dir: str = "outputs/figures",
) -> dict[str, dict]:
    """Run full comparison and produce all outputs.

    Returns dict of {method_name: metrics_dict}.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate both methods
    pf8_metrics = evaluate_peakfinder8(raw_images, labels, pf8_params)
    vit_metrics = evaluate_vit(preprocessed_tensor, labels, vit_model, device)

    results = {"Peakfinder8": pf8_metrics, "ViT": vit_metrics}

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print_comparison_table(results)
    print()

    # Generate plots
    plot_roc_curves(results, save_path=figures_dir / "roc_curves.png")
    plot_pr_curves(results, save_path=figures_dir / "pr_curves.png")
    plot_confusion_matrices(results, save_path=figures_dir / "confusion_matrices.png")
    plot_peak_count_histogram(
        pf8_metrics["peak_counts"],
        labels,
        threshold=pf8_metrics["optimal_threshold"],
        save_path=figures_dir / "pf8_peak_histogram.png",
    )

    print(f"\nFigures saved to {figures_dir}/")
    return results
