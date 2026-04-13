#!/usr/bin/env python3
"""Train the Vision Transformer for hit/miss classification."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.data.dataset import build_dataloaders
from src.vit.train import train_vit


def main():
    parser = argparse.ArgumentParser(description="Train ViT hit finder")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument("--device", default=None, help="Force device (cuda/cpu)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["output_dir"])
    train_path = data_dir / "train.h5"
    val_path = data_dir / "val.h5"

    if not train_path.exists():
        print(f"Error: {train_path} not found. Run generate_data.py first.")
        sys.exit(1)

    vit_cfg = cfg["vit"]
    train_cfg = cfg["training"]

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Build data loaders
    train_loader, val_loader, mean, std = build_dataloaders(
        train_path=train_path,
        val_path=val_path,
        input_size=vit_cfg["input_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    # Save dataset stats for evaluation
    stats_path = data_dir / "stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump({"mean": mean, "std": std}, f)
    print(f"Dataset stats saved to {stats_path}")

    # Train
    model = train_vit(
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=vit_cfg["model_name"],
        pretrained=vit_cfg["pretrained"],
        phase1_epochs=train_cfg["phase1_epochs"],
        phase1_lr=train_cfg["phase1_lr"],
        phase2_epochs=train_cfg["phase2_epochs"],
        phase2_lr=train_cfg["phase2_lr"],
        weight_decay=train_cfg["weight_decay"],
        warmup_epochs=train_cfg["warmup_epochs"],
        patience=train_cfg["patience"],
        mixed_precision=train_cfg["mixed_precision"],
        checkpoint_dir=train_cfg["checkpoint_dir"],
        device=device,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
