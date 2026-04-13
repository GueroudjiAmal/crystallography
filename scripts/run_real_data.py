#!/usr/bin/env python3
"""
End-to-end pipeline for CXIDB 17 real data:
1. Assemble CSPAD panels and build HDF5 dataset
2. Train ViT on real hits + noise-generated misses
3. Run Peakfinder8 on the same data
4. Compare both methods
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.data.cxidb_loader import (
    build_cxidb17_dataset,
    STORE_SIZE,
)
from src.data.dataset import compute_dataset_stats, DiffractionDataset
from src.data.transforms import build_transforms
from src.vit.model import create_vit_model
from src.vit.train import train_vit
from src.classical.peakfinder8 import PeakFinder8
from src.evaluation.metrics import classification_metrics
from src.evaluation.visualize import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrices,
    plot_peak_count_histogram,
    print_comparison_table,
)

from torch.utils.data import DataLoader


def build_real_datasets(cxi_dir: str, output_dir: str, seed: int = 42):
    """Build train/val/test splits from CXIDB 17 data."""
    output_dir = Path(output_dir)
    full_path = output_dir / "cxidb17_full.h5"

    if full_path.exists():
        print(f"Full dataset already exists at {full_path}, skipping generation")
    else:
        build_cxidb17_dataset(cxi_dir, full_path, n_misses_per_hit=1.0, seed=seed)

    # Split into train/val/test (70/15/15)
    rng = np.random.default_rng(seed + 1)

    with h5py.File(full_path, "r") as f:
        n = len(f["labels"])
        labels = f["labels"][:]

    indices = rng.permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train: n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    for name, idx in splits.items():
        split_path = output_dir / f"cxidb17_{name}.h5"
        if split_path.exists():
            print(f"{split_path} already exists, skipping")
            continue

        print(f"Creating {name} split: {len(idx)} images")
        with h5py.File(full_path, "r") as src:
            img_shape = src["images"].shape[1:]  # (H, W)
            split_labels = src["labels"][:][idx]
            split_npeaks = src["n_peaks"][:][idx]

            with h5py.File(split_path, "w") as dst:
                img_ds = dst.create_dataset(
                    "images",
                    shape=(len(idx), *img_shape),
                    dtype="float32",
                    chunks=(1, *img_shape),
                    compression="gzip",
                    compression_opts=4,
                )
                for j, i in enumerate(tqdm(idx, desc=name)):
                    img_ds[j] = src["images"][i]

                dst.create_dataset("labels", data=split_labels)
                dst.create_dataset("n_peaks", data=split_npeaks)

        print(f"  Saved {split_path}")

    return {name: output_dir / f"cxidb17_{name}.h5" for name in splits}


def evaluate_pf8_real(test_path: str, beam_center: tuple[int, int]) -> dict:
    """Run Peakfinder8 on real test data."""
    pf8 = PeakFinder8(
        beam_center=beam_center,
        min_snr=5.0,
        min_pix_count=2,
        max_pix_count=200,
        min_adc=30.0,
        min_res=15,
        max_res=240,
        n_sigma_clip_iterations=5,
        n_peaks_threshold=5,
    )

    with h5py.File(test_path, "r") as f:
        labels = f["labels"][:]
        n = len(labels)

        peak_counts = []
        print("Running Peakfinder8 on test set...")
        for i in tqdm(range(n), desc="PF8"):
            img = f["images"][i]
            result = pf8.find_peaks(img)
            peak_counts.append(result.n_peaks)

    peak_counts = np.array(peak_counts)

    # Find optimal threshold
    from sklearn.metrics import f1_score as f1_fn

    best_f1 = 0.0
    best_t = 10
    for t in range(0, int(peak_counts.max()) + 2):
        preds = (peak_counts >= t).astype(int)
        f1 = f1_fn(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    y_score = peak_counts.astype(float)
    if y_score.max() > y_score.min():
        y_score_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min())
    else:
        y_score_norm = y_score

    y_pred = (peak_counts >= best_t).astype(int)
    metrics = classification_metrics(labels, y_pred, y_score_norm)
    metrics["peak_counts"] = peak_counts
    metrics["optimal_threshold"] = best_t
    return metrics


def evaluate_vit_real(
    test_path: str,
    model: torch.nn.Module,
    mean: float,
    std: float,
    device: torch.device,
    input_size: int = 224,
    batch_size: int = 16,
) -> dict:
    """Run ViT on real test data."""
    import torch.nn.functional as F

    transform = build_transforms(input_size, mean, std, augment=False)
    ds = DiffractionDataset(test_path, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    model = model.to(device)

    all_probs = []
    all_labels = []

    print("Running ViT on test set...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="ViT"):
            images = images.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    y_score = np.array(all_probs)
    y_true = np.array(all_labels)

    # Find optimal threshold
    from sklearn.metrics import f1_score as f1_fn

    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_score >= t).astype(int)
        f1 = f1_fn(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    y_pred = (y_score >= best_t).astype(int)
    metrics = classification_metrics(y_true, y_pred, y_score)
    metrics["optimal_threshold"] = best_t
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run real-data comparison")
    parser.add_argument(
        "--cxi-dir",
        default="data/cxidb-17-run0340",
        help="Directory with CXI files",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase1-epochs", type=int, default=10)
    parser.add_argument("--phase2-epochs", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    output_dir = Path("data/cxidb17_processed")
    figures_dir = Path("outputs/figures_real")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build dataset
    print("\n" + "=" * 60)
    print("Step 1: Building CXIDB 17 dataset")
    print("=" * 60)
    splits = build_real_datasets(args.cxi_dir, str(output_dir))

    # Step 2: Train ViT
    print("\n" + "=" * 60)
    print("Step 2: Training ViT on real data")
    print("=" * 60)

    mean, std = compute_dataset_stats(str(splits["train"]), input_size=224, max_samples=500)
    print(f"Dataset stats: mean={mean:.4f}, std={std:.4f}")

    # Save stats
    with open(output_dir / "stats.yaml", "w") as f:
        yaml.dump({"mean": mean, "std": std}, f)

    train_transform = build_transforms(224, mean, std, augment=True)
    val_transform = build_transforms(224, mean, std, augment=False)

    train_ds = DiffractionDataset(str(splits["train"]), train_transform)
    val_ds = DiffractionDataset(str(splits["val"]), val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    checkpoint_dir = "outputs/checkpoints_real"

    model = train_vit(
        train_loader=train_loader,
        val_loader=val_loader,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        patience=10,
        mixed_precision=(device.type == "cuda"),
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    # Step 3: Evaluate both
    print("\n" + "=" * 60)
    print("Step 3: Evaluating both methods on test set")
    print("=" * 60)

    test_path = str(splits["test"])

    # CSPAD beam center in downsampled image (512x512)
    beam_center = (STORE_SIZE // 2, STORE_SIZE // 2)

    pf8_metrics = evaluate_pf8_real(test_path, beam_center)
    vit_metrics = evaluate_vit_real(test_path, model, mean, std, device, batch_size=args.batch_size)

    results = {"Peakfinder8": pf8_metrics, "ViT": vit_metrics}

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS (CXIDB 17 - Real Data)")
    print("=" * 70)
    print_comparison_table(results)

    # Plots
    plot_roc_curves(results, save_path=figures_dir / "roc_curves_real.png")
    plot_pr_curves(results, save_path=figures_dir / "pr_curves_real.png")
    plot_confusion_matrices(results, save_path=figures_dir / "confusion_matrices_real.png")

    if "peak_counts" in pf8_metrics:
        with h5py.File(test_path, "r") as f:
            labels = f["labels"][:]
        plot_peak_count_histogram(
            pf8_metrics["peak_counts"],
            labels,
            threshold=pf8_metrics["optimal_threshold"],
            save_path=figures_dir / "pf8_peak_histogram_real.png",
        )

    print(f"\nFigures saved to {figures_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
