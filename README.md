# SACR: Structure-Aware Context Refinement for One-Stage Unsupervised Semantic Segmentation

## Installation

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
Data Preparation
Please organize your datasets (VOC, COCO, etc.) under the data/ folder or modify the config files in config/ to point to your dataset path.

## Training

## Step 1: Extract Text Embeddings
First, generate the CLIP text embeddings for your target dataset (e.g., VOC).
```bash
  python utils/prompt_engineering.py --model ViT16 --class-set voc
  # The Text Embeddings will be saved at 'text/voc_ViT16_clip_text.pth' 
  # Options for class-set: voc, context, ade, city, stuff
```
## Step 2: Extract Image-level Multi-label Hypothesis (Pseudo Labels)
Generate the initial pseudo labels using CLIP-CAM or similar techniques.

```bash
    python tools/pseudo_class.py --cfg config/voc_train_ori_cfg.yaml --model SACR
    #The pseudo labels will be saved at 'text/voc_pseudo_label_SACR.json'
```
## Step 3: Train SACR
Train the refinement modules while keeping the CLIP backbone frozen.
```bash
  python tools/train.py --cfg config/voc_train_ori_cfg.yaml
  # Checkpoints will be saved in 'experiments/'
```
## Testing

## Evaluation & Visualization
Evaluate the trained model on the validation set.
```bash
    python tools/test.py --cfg config/voc_test_ori_cfg.yaml --load checkpoints/best_model.pth --save_vis
    # Results (mIoU) will be printed to console
    # Visualization images (if --save_vis is used) will be saved in 'test_run_.../visualization/'
```



