#!/usr/bin/env python3
"""Generate synthetic diffraction datasets for the SFX hit finder PoC."""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from src.data.synthetic import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SFX data")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--difficulty", default=None, help="Override difficulty level"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    difficulty = args.difficulty or data_cfg["difficulty"]
    output_dir = Path(data_cfg["output_dir"])
    image_size = data_cfg["image_size"]
    beam_center = tuple(data_cfg["beam_center"])

    print(f"Generating synthetic SFX data (difficulty={difficulty})")
    print(f"Image size: {image_size}x{image_size}, beam center: {beam_center}")
    print(f"Output dir: {output_dir}")
    print()

    # Training set (balanced)
    print(f"--- Training set ({data_cfg['n_train']} images, balanced) ---")
    generate_dataset(
        n_images=data_cfg["n_train"],
        output_path=output_dir / "train.h5",
        image_size=image_size,
        beam_center=beam_center,
        difficulty=difficulty,
        hit_rate=0.5,
        seed=42,
    )
    print()

    # Validation set (balanced)
    print(f"--- Validation set ({data_cfg['n_val']} images, balanced) ---")
    generate_dataset(
        n_images=data_cfg["n_val"],
        output_path=output_dir / "val.h5",
        image_size=image_size,
        beam_center=beam_center,
        difficulty=difficulty,
        hit_rate=0.5,
        seed=123,
    )
    print()

    # Test set (balanced)
    print(f"--- Test set ({data_cfg['n_test']} images, balanced) ---")
    generate_dataset(
        n_images=data_cfg["n_test"],
        output_path=output_dir / "test.h5",
        image_size=image_size,
        beam_center=beam_center,
        difficulty=difficulty,
        hit_rate=0.5,
        seed=456,
    )
    print()

    # Imbalanced test set (realistic hit rate)
    hit_rate = data_cfg["imbalanced_hit_rate"]
    n_imb = data_cfg["n_test_imbalanced"]
    print(f"--- Imbalanced test set ({n_imb} images, {hit_rate:.0%} hit rate) ---")
    generate_dataset(
        n_images=n_imb,
        output_path=output_dir / "test_imbalanced.h5",
        image_size=image_size,
        beam_center=beam_center,
        difficulty=difficulty,
        hit_rate=hit_rate,
        seed=789,
    )
    print()

    print("Done! All datasets generated.")


if __name__ == "__main__":
    main()
