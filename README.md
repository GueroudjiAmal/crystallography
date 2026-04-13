# SFX Hit Finder: Peakfinder8 vs Vision Transformer

Proof of concept comparing the state-of-the-art classical hit finder (**Peakfinder8** from Cheetah) against a **Vision Transformer (ViT-S/16)** for serial femtosecond crystallography (SFX) hit classification.

Tested on real LCLS CSPAD data from [CXIDB ID 17](https://www.cxidb.org/id-17.html) (Boutet et al., Science 2012 -- lysozyme SFX).

## Key Result

On 784 unseen real diffraction frames:

| Method | Accuracy | Recall | F1 | ROC-AUC | Missed Hits |
|---|---|---|---|---|---|
| Peakfinder8 | 86.9% | 74.9% | 0.851 | 0.801 | 197 / 784 |
| **ViT** | **99.0%** | **98.7%** | **0.990** | **0.999** | **10 / 784** |

The ViT was trained on only 22 labeled examples per class and achieved a **20x reduction in missed hits**.

See [REPORT.md](REPORT.md) for full analysis.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy h5py torch torchvision timm scikit-learn matplotlib tqdm pyyaml
```

## Quick Start -- Synthetic Data

```bash
# Generate synthetic diffraction data (hard difficulty)
python scripts/generate_data.py --config configs/hard.yaml

# Train ViT
python scripts/train_vit.py --config configs/hard.yaml

# Compare PF8 vs ViT
python scripts/run_comparison.py --config configs/hard.yaml
```

## Real Data (CXIDB 17)

Place CXI files from [CXIDB ID 17](https://www.cxidb.org/id-17.html) under `data/cxidb-17-run0340/`.

```bash
# Train ViT on a small subset, evaluate on remaining frames
python scripts/run_real_data.py --cxi-dir data/cxidb17_subset

# Full-scale test on all unseen frames
python scripts/run_full_real_test.py
```

## Project Structure

```
sfx-hit-finder/
├── configs/                # YAML configs (default, hard, small_test)
├── src/
│   ├── data/
│   │   ├── synthetic.py    # Physics-informed diffraction pattern generator
│   │   ├── cxidb_loader.py # CXIDB 17 CSPAD assembler + noise-matched miss generator
│   │   ├── dataset.py      # PyTorch Dataset / DataLoader
│   │   └── transforms.py   # Log-scale, resize, normalize, augmentations
│   ├── classical/
│   │   ├── peakfinder8.py  # Peakfinder8 algorithm (8-step, NumPy/SciPy)
│   │   └── hit_finder.py   # Batch evaluation + threshold sweep
│   ├── vit/
│   │   ├── model.py        # ViT-S/16 via timm (1-channel, ImageNet-21k pretrained)
│   │   └── train.py        # Two-phase training (head-only, then full fine-tuning)
│   └── evaluation/
│       ├── metrics.py      # Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, timing
│       ├── compare.py      # Side-by-side benchmark
│       └── visualize.py    # ROC/PR curves, confusion matrices, peak histograms
├── scripts/
│   ├── generate_data.py    # Generate synthetic datasets
│   ├── train_vit.py        # Train ViT from config
│   ├── run_comparison.py   # Evaluate both methods on synthetic test set
│   ├── run_real_data.py    # Train + evaluate on CXIDB 17 (small subset)
│   └── run_full_real_test.py  # Full-scale evaluation on all unseen CXI frames
└── outputs/                # Checkpoints, figures, results (git-ignored)
```

## How It Works

**Peakfinder8** estimates per-ring background via iterative sigma-clipping, finds connected-component peaks above threshold, and classifies by peak count. It requires manual tuning of 8+ parameters per experiment.

**ViT** takes the full image (log-scaled, resized to 224x224), splits it into 16x16 patches, and uses transformer self-attention to learn the spatial signature of diffraction. Pretrained on ImageNet-21k, it transfers to crystallography with minimal fine-tuning.

## References

1. Boutet, S. et al. "High-Resolution Protein Structure Determination by Serial Femtosecond Crystallography." *Science* 337, 362-364 (2012).
2. Barty, A. et al. "Cheetah: software for high-throughput reduction and analysis of serial femtosecond X-ray diffraction data." *J. Appl. Cryst.* 47, 1118-1131 (2014).
3. Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR* (2021).

## License

MIT License. See [LICENSE](LICENSE).
