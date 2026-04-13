"""
Two-phase ViT training loop for hit/miss classification.

Phase 1: Train classification head only (backbone frozen).
Phase 2: Full fine-tuning with cosine annealing and warmup.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score

from src.vit.model import create_vit_model, freeze_backbone, unfreeze_all


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Validate model. Returns (loss, accuracy, f1)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="binary")
    return total_loss / n, float(acc), float(f1)


def train_vit(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str = "vit_small_patch16_224",
    pretrained: bool = True,
    phase1_epochs: int = 10,
    phase1_lr: float = 1e-3,
    phase2_epochs: int = 40,
    phase2_lr: float = 1e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    patience: int = 10,
    mixed_precision: bool = True,
    checkpoint_dir: str = "outputs/checkpoints",
    device: torch.device | None = None,
) -> nn.Module:
    """Full two-phase training pipeline.

    Returns the best model (by validation F1).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = create_vit_model(model_name, pretrained=pretrained)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    best_f1 = 0.0
    best_state = None
    epochs_without_improvement = 0

    # ── Phase 1: Head-only training ──────────────────────────
    print("=" * 60)
    print("Phase 1: Training classification head (backbone frozen)")
    print("=" * 60)

    freeze_backbone(model)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase1_lr,
        weight_decay=weight_decay,
    )

    for epoch in range(1, phase1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:3d}/{phase1_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

    # ── Phase 2: Full fine-tuning ────────────────────────────
    print()
    print("=" * 60)
    print("Phase 2: Full fine-tuning (all layers)")
    print("=" * 60)

    unfreeze_all(model)
    optimizer = AdamW(model.parameters(), lr=phase2_lr, weight_decay=weight_decay)

    # Warmup + cosine annealing
    effective_warmup = min(warmup_epochs, phase2_epochs - 1)
    cosine_t_max = max(phase2_epochs - effective_warmup, 1)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=max(effective_warmup, 1)
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=cosine_t_max
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[effective_warmup],
    )

    epochs_without_improvement = 0

    for epoch in range(1, phase2_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{phase2_epochs} | "
            f"lr={lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(best_state, checkpoint_dir / "best_vit.pt")
    print(f"\nBest validation F1: {best_f1:.4f}")
    print(f"Checkpoint saved to {checkpoint_dir / 'best_vit.pt'}")

    return model
