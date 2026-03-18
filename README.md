# Drywall Defect Segmentation — CLIPSeg + SAM

Prompt-driven wall defect detection pipeline using **CLIPSeg** (CLIP-based image segmentation) and **Segment Anything Model (SAM)** from Meta. Fine-tuned to detect and segment drywall cracks and joints from image datasets.

## Project Summary

1. **Project:** Developed a computer vision pipeline to detect and segment structural damage like wall cracks and drywall joints.
2. **Challenge:** Identifying fine-grained features in complex images without massive manually labeled datasets.
3. **Solution:** Combined OpenAI's CLIP for zero-shot detection with Meta's SAM for precise pixel-level mask generation.
4. **Outcome:** Built a successful, modular system that accurately segments cracks and generates comprehensive evaluation metrics.
5. **Takeaway:** Integrating foundation models provides highly adaptable and powerful solutions even when labeled data is scarce.

## Overview

- Uses `CLIPSegForImageSegmentation` from HuggingFace Transformers for text-prompted segmentation
- SAM mask generator for high-quality region proposals
- Custom `PromptSegDataset` for loading image/mask pairs with text prompts
- Training, evaluation, and visual report generation scripts

## Structure

| File | Description |
|------|-------------|
| `train.py` | Fine-tuning CLIPSeg on drywall dataset |
| `sam_mask_generator.py` | SAM-based mask proposals |
| `metrics.py` | Evaluation metrics (IoU, precision, recall) |
| `report.py` | Visual report generation |
| `split.py` | Train/val dataset splitting |
| `requirements.txt` | Dependencies |

## Setup

```bash
pip install -r requirements.txt
python train.py
```

## Tech Stack

`PyTorch` · `HuggingFace Transformers` · `CLIPSeg` · `Segment Anything (SAM)` · `OpenCV` · `Python`