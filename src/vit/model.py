"""
Vision Transformer model for diffraction image hit classification.

Uses timm's vit_small_patch16_224 pretrained on ImageNet-21k,
adapted for single-channel grayscale input.
"""

import timm
import torch.nn as nn


def create_vit_model(
    model_name: str = "vit_small_patch16_224",
    pretrained: bool = True,
    in_chans: int = 1,
    num_classes: int = 2,
) -> nn.Module:
    """Create a ViT model for binary hit/miss classification.

    When in_chans=1 and pretrained=True, timm sums the 3 RGB channel
    weights of the patch embedding into a single channel, providing a
    reasonable initialization for grayscale inputs.

    Args:
        model_name: timm model identifier.
        pretrained: Whether to load ImageNet-21k pretrained weights.
        in_chans: Number of input channels (1 for grayscale).
        num_classes: Number of output classes (2 for hit/miss).

    Returns:
        Configured ViT model.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=num_classes,
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classification head."""
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
