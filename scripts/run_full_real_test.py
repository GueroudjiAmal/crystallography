#!/usr/bin/env python3
"""
Full-scale real-data evaluation.

Uses the ViT model trained on 32 frames (from run_real_data.py) and evaluates
both PF8 and ViT on the remaining 784 unseen CXI frames.

All 816 CXIDB 17 frames are pre-identified hits (by Cheetah).
Misses are generated from the real detector noise profile.
The 32 training frames are excluded from the test set.
"""

import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import torch.nn.functional as Fn
import yaml
from tqdm import tqdm

from src.data.cxidb_loader import (
    assemble_cspad,
    extract_noise_profile,
    generate_miss_from_noise,
    STORE_SIZE,
)
from src.data.transforms import build_transforms
from src.vit.model import create_vit_model
from src.classical.peakfinder8 import PeakFinder8
from src.evaluation.metrics import classification_metrics
from src.evaluation.visualize import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrices,
    plot_peak_count_histogram,
    print_comparison_table,
)


def downsample(image: np.ndarray, size: int = STORE_SIZE) -> np.ndarray:
    t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    t = Fn.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def main():
    device = torch.device("cpu")
    figures_dir = Path("outputs/figures_full_real")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Identify train vs test files ────────────────────────
    all_cxi = sorted(glob.glob("data/cxidb-17-run0340/**/*.cxi", recursive=True))
    train_cxi = set(Path(p).name for p in glob.glob("data/cxidb17_subset/*.cxi"))
    test_cxi = [p for p in all_cxi if Path(p).name not in train_cxi]

    print(f"Total CXI files: {len(all_cxi)}")
    print(f"Training files (excluded): {len(train_cxi)}")
    print(f"Test files (unseen): {len(test_cxi)}")

    # ── Load trained ViT ────────────────────────────────────
    ckpt_path = Path("outputs/checkpoints_real/best_vit.pt")
    stats_path = Path("data/cxidb17_processed/stats.yaml")

    if not ckpt_path.exists() or not stats_path.exists():
        print("Error: Run scripts/run_real_data.py first to train the ViT.")
        sys.exit(1)

    with open(stats_path) as f:
        stats = yaml.safe_load(f)
    mean, std = stats["mean"], stats["std"]

    model = create_vit_model(
        model_name="vit_small_patch16_224",
        pretrained=False, in_chans=1, num_classes=2,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded ViT from {ckpt_path}")

    # ── Extract noise profile for miss generation ───────────
    print("Extracting noise profile from test files...")
    rng = np.random.default_rng(999)
    mean_img, std_img = extract_noise_profile(test_cxi, n_samples=50, rng=rng)
    mean_img_ds = downsample(mean_img, STORE_SIZE)
    std_img_ds = downsample(std_img, STORE_SIZE)

    # ── Build test set: all unseen hits + equal number of misses ──
    n_hits = len(test_cxi)
    n_misses = n_hits
    n_total = n_hits + n_misses

    print(f"\nBuilding test set: {n_hits} real hits + {n_misses} noise misses = {n_total}")

    labels = np.array([1] * n_hits + [0] * n_misses, dtype=np.uint8)
    order = rng.permutation(n_total)
    labels = labels[order]

    # ── Prepare PF8 ─────────────────────────────────────────
    beam_center = (STORE_SIZE // 2, STORE_SIZE // 2)
    pf8 = PeakFinder8(
        beam_center=beam_center, min_snr=5.0, min_pix_count=2,
        max_pix_count=200, min_adc=30.0, min_res=15, max_res=240,
        n_sigma_clip_iterations=5, n_peaks_threshold=5,
    )

    # ── Prepare ViT transform ──────────────────────────────
    transform = build_transforms(224, mean, std, augment=False)

    # ── Evaluate both methods in a single pass ──────────────
    pf8_peak_counts = []
    vit_probs = []

    print("Evaluating (this may take a while on CPU)...")
    for out_idx in tqdm(range(n_total), desc="Testing"):
        orig_idx = order[out_idx]
        is_hit = orig_idx < n_hits

        if is_hit:
            path = test_cxi[orig_idx]
            with h5py.File(path, "r") as f:
                img_full = assemble_cspad(f)
            img = downsample(img_full, STORE_SIZE)
        else:
            img = generate_miss_from_noise(mean_img_ds, std_img_ds, rng)

        # PF8
        result = pf8.find_peaks(img)
        pf8_peak_counts.append(result.n_peaks)

        # ViT
        processed = transform(img.astype(np.float32))
        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            logits = model(tensor)
        prob = Fn.softmax(logits, dim=1)[0, 1].item()
        vit_probs.append(prob)

    pf8_peak_counts = np.array(pf8_peak_counts)
    vit_probs = np.array(vit_probs)

    # ── Compute metrics ─────────────────────────────────────
    # PF8: find optimal threshold
    from sklearn.metrics import f1_score as f1_fn

    best_f1_pf8, best_t_pf8 = 0.0, 5
    for t in range(0, int(pf8_peak_counts.max()) + 2):
        preds = (pf8_peak_counts >= t).astype(int)
        f1 = f1_fn(labels, preds, zero_division=0)
        if f1 > best_f1_pf8:
            best_f1_pf8 = f1
            best_t_pf8 = t

    pf8_score = pf8_peak_counts.astype(float)
    if pf8_score.max() > pf8_score.min():
        pf8_score_norm = (pf8_score - pf8_score.min()) / (pf8_score.max() - pf8_score.min())
    else:
        pf8_score_norm = pf8_score

    pf8_pred = (pf8_peak_counts >= best_t_pf8).astype(int)
    pf8_metrics = classification_metrics(labels, pf8_pred, pf8_score_norm)
    pf8_metrics["peak_counts"] = pf8_peak_counts
    pf8_metrics["optimal_threshold"] = best_t_pf8

    # ViT: find optimal threshold
    best_f1_vit, best_t_vit = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (vit_probs >= t).astype(int)
        f1 = f1_fn(labels, preds, zero_division=0)
        if f1 > best_f1_vit:
            best_f1_vit = f1
            best_t_vit = t

    vit_pred = (vit_probs >= best_t_vit).astype(int)
    vit_metrics = classification_metrics(labels, vit_pred, vit_probs)
    vit_metrics["optimal_threshold"] = best_t_vit

    results = {"Peakfinder8": pf8_metrics, "ViT": vit_metrics}

    # ── Print results ───────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"FULL REAL-DATA RESULTS — CXIDB 17 ({n_hits} hits + {n_misses} misses)")
    print("=" * 80)
    table = print_comparison_table(results)
    print(f"\nPF8 optimal threshold: {best_t_pf8} peaks")
    print(f"ViT optimal threshold: {best_t_vit:.2f}")

    # ── Save results to file ────────────────────────────────
    with open(figures_dir / "results.txt", "w") as f:
        f.write(f"FULL REAL-DATA RESULTS — CXIDB 17\n")
        f.write(f"Test set: {n_hits} real hits + {n_misses} noise misses\n")
        f.write(f"ViT trained on: 32 frames (22 train + 9 val)\n\n")
        f.write(table + "\n\n")
        f.write(f"PF8 optimal peak threshold: {best_t_pf8}\n")
        f.write(f"ViT optimal probability threshold: {best_t_vit:.2f}\n\n")
        f.write(f"PF8 confusion matrix:\n{pf8_metrics['confusion_matrix']}\n\n")
        f.write(f"ViT confusion matrix:\n{vit_metrics['confusion_matrix']}\n\n")

        # Detailed error analysis
        pf8_fn_idx = np.where((pf8_pred == 0) & (labels == 1))[0]
        pf8_fp_idx = np.where((pf8_pred == 1) & (labels == 0))[0]
        vit_fn_idx = np.where((vit_pred == 0) & (labels == 1))[0]
        vit_fp_idx = np.where((vit_pred == 1) & (labels == 0))[0]

        f.write(f"PF8 false negatives (missed hits): {len(pf8_fn_idx)}\n")
        f.write(f"PF8 false positives: {len(pf8_fp_idx)}\n")
        f.write(f"ViT false negatives (missed hits): {len(vit_fn_idx)}\n")
        f.write(f"ViT false positives: {len(vit_fp_idx)}\n")

    # ── Plots ───────────────────────────────────────────────
    plot_roc_curves(results, save_path=figures_dir / "roc_curves.png")
    plot_pr_curves(results, save_path=figures_dir / "pr_curves.png")
    plot_confusion_matrices(results, save_path=figures_dir / "confusion_matrices.png")
    plot_peak_count_histogram(
        pf8_peak_counts, labels,
        threshold=best_t_pf8,
        save_path=figures_dir / "pf8_peak_histogram.png",
    )

    print(f"\nFigures and results saved to {figures_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
