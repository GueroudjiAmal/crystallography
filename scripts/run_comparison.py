#!/usr/bin/env python3
"""Run side-by-side comparison of Peakfinder8 and ViT on test data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.data.transforms import build_transforms
from src.vit.model import create_vit_model
from src.evaluation.compare import run_comparison


def load_test_data(
    test_path: str,
    mean: float,
    std: float,
    input_size: int = 224,
) -> tuple[list[np.ndarray], torch.Tensor, np.ndarray]:
    """Load test data for both methods.

    Returns:
        raw_images: list of raw uint16 arrays (for PF8)
        preprocessed_tensor: (N, 1, 224, 224) tensor (for ViT)
        labels: ground-truth labels
    """
    transform = build_transforms(input_size, mean, std, augment=False)

    with h5py.File(test_path, "r") as f:
        labels = f["labels"][:]
        n = len(labels)

        raw_images = []
        preprocessed = []

        for i in tqdm(range(n), desc="Loading test data"):
            img = f["images"][i]
            raw_images.append(img)

            processed = transform(img.astype(np.float32))
            preprocessed.append(
                torch.from_numpy(processed).unsqueeze(0).float()
            )

    preprocessed_tensor = torch.stack(preprocessed)
    return raw_images, preprocessed_tensor, labels


def main():
    parser = argparse.ArgumentParser(description="Compare PF8 vs ViT")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--test-set", default="test",
        choices=["test", "test_imbalanced"],
        help="Which test set to use",
    )
    parser.add_argument("--device", default=None, help="Force device (cuda/cpu)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["output_dir"])
    test_path = data_dir / f"{args.test_set}.h5"
    stats_path = data_dir / "stats.yaml"
    checkpoint_path = Path(cfg["training"]["checkpoint_dir"]) / "best_vit.pt"

    for p, name in [
        (test_path, "test data"),
        (stats_path, "dataset stats"),
        (checkpoint_path, "ViT checkpoint"),
    ]:
        if not p.exists():
            print(f"Error: {name} not found at {p}")
            print("Run generate_data.py and train_vit.py first.")
            sys.exit(1)

    with open(stats_path) as f:
        stats = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    print(f"Test set: {test_path}")

    # Load test data
    raw_images, preprocessed_tensor, labels = load_test_data(
        str(test_path),
        mean=stats["mean"],
        std=stats["std"],
        input_size=cfg["vit"]["input_size"],
    )

    # Load trained ViT
    vit_model = create_vit_model(
        model_name=cfg["vit"]["model_name"],
        pretrained=False,
        in_chans=cfg["vit"]["in_chans"],
        num_classes=cfg["vit"]["num_classes"],
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vit_model.load_state_dict(state_dict)
    print("Loaded ViT checkpoint")

    # Build PF8 params
    pf8_cfg = cfg["peakfinder8"]
    pf8_params = {
        "beam_center": tuple(cfg["data"]["beam_center"]),
        "min_snr": pf8_cfg["min_snr"],
        "min_pix_count": pf8_cfg["min_pix_count"],
        "max_pix_count": pf8_cfg["max_pix_count"],
        "min_adc": pf8_cfg["min_adc"],
        "min_res": pf8_cfg["min_res"],
        "max_res": pf8_cfg["max_res"],
        "n_sigma_clip_iterations": pf8_cfg["n_sigma_clip_iterations"],
        "n_peaks_threshold": pf8_cfg["n_peaks_threshold"],
    }

    # Run comparison
    results = run_comparison(
        raw_images=raw_images,
        preprocessed_tensor=preprocessed_tensor,
        labels=labels,
        vit_model=vit_model,
        pf8_params=pf8_params,
        device=device,
        figures_dir=cfg["evaluation"]["figures_dir"],
    )

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
