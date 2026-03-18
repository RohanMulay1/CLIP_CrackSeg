# Drywall Defect Segmentation — CLIPSeg + SAM

Prompt-driven wall defect detection pipeline using **CLIPSeg** (CLIP-based image segmentation) and **Segment Anything Model (SAM)** from Meta. Fine-tuned to detect and segment drywall cracks and joints from image datasets.

## Overview

- Uses `CLIPSegForImageSegmentation` from HuggingFace Transformers for text-prompted segmentation
- SAM mask generator for high-quality region proposals
- Custom `PromptSegDataset` for loading image/mask pairs with text prompts
- Training, evaluation, and visual report generation scripts

## Repository Structure

```
CLIP_CrackSeg/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── train.py              # Fine-tuning CLIPSeg on drywall dataset
│   ├── sam_mask_generator.py # SAM-based mask proposals
│   ├── metrics.py            # Training metric visualisation
│   ├── report.py             # Visual report generation
│   └── split.py              # Train/val/test dataset splitting
├── data/
│   ├── drywall/              # Drywall-Join-Detect dataset
│   └── wall-crack/           # Wall-crack dataset
├── outputs/                  # Generated images and model checkpoints
└── docs/
    └── Origin_Assignment_RohanMulay.pdf
```

## Setup

```bash
pip install -r requirements.txt
```

### 1. Prepare the datasets

Split the raw training images into train / val / test splits:

```bash
python src/split.py
```

### 2. Generate SAM masks

```bash
python src/sam_mask_generator.py
```

### 3. Train

```bash
python src/train.py
```

### 4. Evaluate & generate report

```bash
python src/report.py
```

Outputs (images and checkpoints) are written to the `outputs/` directory.

## Tech Stack

`PyTorch` · `HuggingFace Transformers` · `CLIPSeg` · `Segment Anything (SAM)` · `OpenCV` · `Python`