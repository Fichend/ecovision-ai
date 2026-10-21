# EcoVision AI MobileNetV2 baseline report

Run date: 2026-10-21 (Asia/Hong_Kong)

## Outcome

The selected MobileNetV2 checkpoint achieved **92.33% test accuracy** and
**90.11% test macro-F1** on a held-out, group-aware TrashNet split. The test
set was evaluated once after choosing the checkpoint with the highest
validation macro-F1.

| Metric | Result |
| --- | ---: |
| Best validation macro-F1 | 88.70% |
| Test accuracy | 92.33% |
| Test macro-F1 | 90.11% |
| Test macro precision | 89.47% |
| Test macro recall | 91.23% |

## Per-class test performance

| Class | Precision | Recall | F1 | Test samples |
| --- | ---: | ---: | ---: | ---: |
| Cardboard | 98.33% | 98.33% | 98.33% | 60 |
| Glass | 96.97% | 85.33% | 90.78% | 75 |
| Metal | 88.06% | 96.72% | 92.19% | 61 |
| Paper | 97.67% | 94.38% | 96.00% | 89 |
| Plastic | 90.41% | 91.67% | 91.03% | 72 |
| Trash | 65.38% | 80.95% | 72.34% | 21 |

The `trash` class is the main weakness. It has only 137 original images and 21
test samples after cleaning, so its metrics have substantially higher
uncertainty than the larger classes.

## Data controls

- Official source images discovered: 2,527.
- Six byte-identical images with conflicting labels were excluded in full.
- Remaining images: 2,521.
- Same-label near-duplicates were grouped with a 64-bit difference hash and a
  Hamming-distance threshold of four.
- Near-duplicate cross-split audit: zero pairs at thresholds zero, two, and
  four.
- Split target: 70% train, 15% validation, 15% test with seed 42.
- Actual sizes: 1,765 train, 378 validation, and 378 test.

An earlier ordinary stratified split achieved 91.03% accuracy and 90.36%
macro-F1, but its audit found 39 near-duplicate pairs crossing partitions. It
is retained only as a comparison and must not be treated as the primary result.

## Training configuration

- Architecture: MobileNetV2 with ImageNet pretrained weights.
- Output classes: six.
- Device: Apple MPS.
- Epochs: 10.
- Feature extractor frozen for two epochs, then fine-tuned.
- Optimizer: AdamW.
- Initial classifier learning rate: 0.001.
- Fine-tuning backbone learning rate: 0.0001.
- Batch size: 32.
- Loss: inverse-frequency class-weighted cross entropy.
- Model selection: validation macro-F1.

## Limitations

TrashNet is small, imbalanced, and photographed mostly against clean white
backgrounds. These results do not establish performance on mixed waste,
partially occluded items, unusual lighting, composite materials, or real Hong
Kong and mainland China recycling-bin scenes. External validation images are
required before making real-world performance claims.

## Attribution

Dataset: Gary Thung and Mindy Yang, TrashNet.

- https://github.com/garythung/trashnet
- https://huggingface.co/datasets/garythung/trashnet
