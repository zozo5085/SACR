# SACR: Structure-Aware Context Refinement for One-Stage Unsupervised Semantic Segmentation

This repository contains the official implementation of the paper:
**"Structure-Aware Context Refinement for One-Stage Unsupervised Semantic Segmentation with Frozen CLIP"**.

## Introduction
SACR is a lightweight one-stage framework that enhances frozen CLIP-ViT features for unsupervised semantic segmentation. It introduces Coordinate Attention, Zero-Initialized Mini-ASPP, and Edge Guidance to improve boundary quality without distillation.

## Installation

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
Data Preparation
Please organize your datasets (VOC, COCO, etc.) under the data/ folder or modify the config files in config/ to point to your dataset path.

Usage
Training
Run the training script:

Bash

python tools/train.py --cfg config/voc_train_ori_cfg.yaml
Testing
Bash

python tools/test.py --cfg config/voc_test_ori_cfg.yaml --load checkpoints/best_model.pth --save_vis
Generating Pseudo Labels
Bash

python tools/pseudo_class.py --cfg config/voc_train_ori_cfg.yaml
