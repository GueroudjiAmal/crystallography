"""
Visualization utilities for comparing hit finder methods.

Produces ROC curves, PR curves, confusion matrices, example galleries,
and ViT attention maps.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_roc_curves(
    results: dict[str, dict],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot overlaid ROC curves for multiple methods.

    Args:
        results: {method_name: metrics_dict} where metrics_dict contains 'roc_curve'.
        save_path: If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    for name, m in results.items():
        if "roc_curve" in m:
            auc = m.get("roc_auc", 0)
            ax.plot(
                m["roc_curve"]["fpr"],
                m["roc_curve"]["tpr"],
                linewidth=2,
                label=f"{name} (AUC={auc:.3f})",
            )

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve Comparison", fontsize=15)
    ax.legend(fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pr_curves(
    results: dict[str, dict],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot overlaid Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, m in results.items():
        if "pr_curve" in m:
            auc = m.get("pr_auc", 0)
            ax.plot(
                m["pr_curve"]["recall"],
                m["pr_curve"]["precision"],
                linewidth=2,
                label=f"{name} (AP={auc:.3f})",
            )

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve Comparison", fontsize=15)
    ax.legend(fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrices(
    results: dict[str, dict],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot confusion matrices side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, m) in zip(axes, results.items()):
        cm = m["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{name}", fontsize=13)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Miss", "Hit"])
        ax.set_yticklabels(["Miss", "Hit"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrices", fontsize=15, y=1.02)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_example_gallery(
    images: list[np.ndarray],
    true_labels: list[int],
    pred_labels: dict[str, list[int]],
    n_examples: int = 4,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot example images for TP, FP, FN, TN categories for each method."""
    methods = list(pred_labels.keys())
    categories = ["TP", "FP", "FN", "TN"]

    fig = plt.figure(figsize=(4 * n_examples, 4 * len(methods) * len(categories) // 2))
    gs = gridspec.GridSpec(len(methods) * 2, n_examples, figure=fig)

    true_arr = np.array(true_labels)

    for m_idx, method in enumerate(methods):
        pred_arr = np.array(pred_labels[method])

        tp_idx = np.where((pred_arr == 1) & (true_arr == 1))[0]
        fp_idx = np.where((pred_arr == 1) & (true_arr == 0))[0]
        fn_idx = np.where((pred_arr == 0) & (true_arr == 1))[0]
        tn_idx = np.where((pred_arr == 0) & (true_arr == 0))[0]

        for cat_idx, (cat_name, indices) in enumerate(
            [("TP", tp_idx), ("FP", fp_idx), ("FN", fn_idx), ("TN", tn_idx)]
        ):
            if cat_idx >= 2:
                continue  # Show only TP and FP rows per method for space

            row = m_idx * 2 + cat_idx
            sel = indices[:n_examples]
            for j, idx in enumerate(sel):
                ax = fig.add_subplot(gs[row, j])
                ax.imshow(np.log1p(images[idx].astype(np.float32)), cmap="viridis")
                ax.set_title(f"{method} {cat_name}", fontsize=9)
                ax.axis("off")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_peak_count_histogram(
    peak_counts: np.ndarray,
    labels: np.ndarray,
    threshold: int | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot histogram of detected peak counts for hits vs misses."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        peak_counts[labels == 1],
        bins=range(0, int(peak_counts.max()) + 2),
        alpha=0.6, label="Hits", color="tab:blue",
    )
    ax.hist(
        peak_counts[labels == 0],
        bins=range(0, int(peak_counts.max()) + 2),
        alpha=0.6, label="Misses", color="tab:orange",
    )

    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
                    label=f"Threshold = {threshold}")

    ax.set_xlabel("Number of detected peaks", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Peakfinder8: Peak Count Distribution", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def print_comparison_table(results: dict[str, dict]) -> str:
    """Print a formatted comparison table and return it as a string."""
    header = f"{'Method':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    has_auc = any("roc_auc" in m for m in results.values())
    has_timing = any("timing" in m for m in results.values())
    if has_auc:
        header += f" {'ROC-AUC':>10} {'PR-AUC':>10}"
    if has_timing:
        header += f" {'ms/img':>10} {'img/s':>10}"

    lines = [header, "-" * len(header)]

    for name, m in results.items():
        line = (
            f"{name:<20} "
            f"{m['accuracy']:>10.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f}"
        )
        if has_auc:
            line += f" {m.get('roc_auc', 0):>10.4f} {m.get('pr_auc', 0):>10.4f}"
        if has_timing:
            t = m.get("timing", {})
            line += f" {t.get('mean_ms', 0):>10.2f} {t.get('throughput_per_sec', 0):>10.1f}"
        lines.append(line)

    table = "\n".join(lines)
    print(table)
    return table
