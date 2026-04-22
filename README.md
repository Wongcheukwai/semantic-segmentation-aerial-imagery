# Semantic Segmentation for Roof and Solar Panel Detection

A multi-label semantic segmentation solution for detecting roofs and solar panels in aerial imagery using PyTorch U-Net architectures with transfer learning.

## Problem Overview

Given a dataset of aerial/satellite images, the task is to **classify every pixel** into one of four possible states:

| Pixel State | Roof? | Solar Panel? |
|---|---|---|
| Background (e.g. grass, road, tree) | No | No |
| Roof only | Yes | No |
| Solar panel only | No | Yes |
| Both roof and solar panel | Yes | Yes |

This is fundamentally a **multi-label** problem, not a multi-class one. A single pixel can belong to both "roof" and "solar panel" simultaneously (since solar panels are installed on roofs). This distinction matters because:

- **Multi-class** (softmax) forces mutually exclusive categories — a pixel cannot be both roof and solar
- **Multi-label** (sigmoid) allows independent predictions per category — a pixel can be roof=1, solar=1

The dataset uses sparse **16x16 grid annotations**: only ~0.4% of pixels in each image have labels (the rest are marked as -1, meaning "unknown"). Among the labelled pixels, solar panel positives are extremely rare — only **0.0004%** of all pixels across the dataset, with **89% of images having zero solar-positive pixels**.

## Architecture

### Two Models

I implemented and compared two architectures:

**1. TinyUNet** — A simplified U-Net built from scratch
- 4-level encoder-decoder with skip connections
- ~17M parameters
- Serves as the baseline to show the model can learn from this data

**2. ResNetUNet** — A U-Net with a pretrained ResNet34 encoder
- ImageNet-pretrained ResNet34 as the encoder (transfer learning)
- Custom decoder with skip connections from each encoder stage
- ~24.5M parameters
- Transfer learning provides a strong starting point, especially with limited labelled data

Both models output **2 independent channels** with sigmoid activation:
- Channel 0: Roof probability (per pixel)
- Channel 1: Solar panel probability (per pixel)

### Two Loss Functions

**1. Masked BCE (Binary Cross-Entropy)**
- Standard BCE loss, but computed only on valid pixels (where label != -1)
- Invalid pixels are excluded via a valid mask, so they contribute zero gradient
- Channel weights [1.0, 2.0] give solar panels higher weight due to sparsity

**2. Masked BCE + Dice (Combined)**
- BCE provides stable per-pixel gradients
- Dice loss directly optimises region overlap (F1/Dice score), which is crucial for sparse targets like solar panels where BCE alone tends to predict all-zero (since that minimises per-pixel loss when 99%+ of pixels are negative)
- Combined as 0.5 * BCE + 0.5 * Dice

### The Solar Panel Challenge

Solar panel detection is extremely difficult on this dataset due to:
- **0.0004%** positive pixel ratio across the entire dataset
- **89% of images** contain zero solar-positive labels
- With standard BCE loss, predicting "no solar everywhere" achieves 99.89% accuracy — the model has no incentive to predict positives

The TinyUNet + BCE baseline **completely fails** on solar (all metrics = 0), confirming this is a data imbalance problem, not a model capacity problem. The ResNetUNet + BCE+Dice combination successfully detects solar panels (IoU = 0.60) because:
1. The pretrained encoder already understands visual features relevant to solar panels
2. Dice loss penalises the all-zero prediction by directly measuring region overlap

## Key Technical Decisions

### 1. Handling Invalid Pixels (-1 labels)
```python
# -1: Unknown/unlabeled (excluded from loss computation)
# 0: Explicitly negative (no roof / no solar)
# >=1: Positive (converted to binary 1)

valid_mask = (original_mask != -1)
binary_mask = (original_mask >= 1).float()
```

### 2. Data Preprocessing
- Resize to 512x512 for training (balances speed and detail)
- Standard ImageNet normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Augmentations: random horizontal/vertical flips and colour jitter

### 3. Training Strategy
- AdamW optimiser with weight decay (1e-4)
- Cosine annealing learning rate schedule
- Best model checkpoint saved based on validation loss

## Project Structure

```
semantic-segmentation-aerial/
├── README.md
├── requirements.txt
├── run.py                  # Main pipeline entry point
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Data loading, preprocessing, augmentation
│   ├── model.py            # TinyUNet and ResNetUNet architectures
│   ├── metrics.py          # Evaluation metrics (accuracy, precision, recall, F1, IoU)
│   ├── train.py            # Training loop, loss functions
│   ├── evaluate.py         # Model evaluation on validation set
│   └── visualise.py        # Prediction visualisation and training plots
└── MLtestdata/             # Dataset (images + sparse mask annotations)
```

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Experiment 1: TinyUNet + BCE (Baseline)
```bash
python run.py --model tinyunet --loss bce --epochs 10 --batch-size 8 --img-size 512
```

### Experiment 2: ResNetUNet + BCE+Dice (Improved)
```bash
python run.py --model resnetunet --loss bce_dice --epochs 15 --batch-size 8 --img-size 512
```

## Results

### Experiment Comparison

| Metric | TinyUNet + BCE | ResNetUNet + BCE+Dice |
|---|---|---|
| **Roof Accuracy** | 0.910 | **0.940** |
| **Roof Precision** | 0.819 | **0.846** |
| **Roof Recall** | 0.569 | **0.766** |
| **Roof F1/Dice** | 0.671 | **0.804** |
| **Roof IoU** | 0.505 | **0.672** |
| **Solar Precision** | 0.000 | **0.728** |
| **Solar Recall** | 0.000 | **0.769** |
| **Solar F1/Dice** | 0.000 | **0.748** |
| **Solar IoU** | 0.000 | **0.597** |

Key observations:
- **Loss decreases consistently** across all epochs in both experiments, confirming the models are learning
- **Pretrained encoder** (ResNet34) significantly boosts roof detection: IoU improves from 0.505 to 0.672
- **BCE+Dice loss** is critical for solar panel detection: TinyUNet+BCE produces all-zero predictions for solar, while ResNetUNet+BCE+Dice achieves IoU=0.597
- The combination of transfer learning + Dice loss solves the solar panel sparsity problem

### Training and Evaluation

During training, the models save checkpoints, training logs, and validation metrics. Model results and training curves can be generated by running the scripts above.

## What I Would Do With More Time and Data

### Priority 1: Addressing Solar Panel Sparsity Further

1. **Positive-biased sampling** — oversample images that contain solar panel labels so every batch has positive examples
2. **Per-class threshold tuning** — find the optimal prediction threshold per channel by maximising F1 on the validation set, rather than using a fixed 0.5
3. **Focal loss** — down-weight easy negatives to focus the model on hard boundary pixels

### Priority 2: Better Training and Augmentation

1. **Longer training** (50-100 epochs) with early stopping on validation IoU rather than loss
2. **Stronger augmentations** — RandomRotate90, elastic deformations, random crops (train on 512x512 patches from 896x896 images)
3. **Cosine warmup** — ramp learning rate from 0 over the first few epochs to avoid destroying pretrained weights
4. **Mixed precision (AMP)** — nearly free 2x speedup on GPU

### Priority 3: Scaling Up

1. **Data quality pipeline** — systematic review of label quality, active learning to prioritise the most informative samples for annotation
2. **Stronger architectures** — ConvNeXt or EfficientNet encoders via `segmentation_models_pytorch`, attention mechanisms, Feature Pyramid Networks
3. **Instance segmentation** — separate individual roofs and panels (Mask R-CNN) for downstream analytics like panel counting
