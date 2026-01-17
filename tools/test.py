import argparse
import os
import sys
from datetime import datetime
import cv2
import torch
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())

from config.configs import cfg_from_file
from model.model import SACR
from utils.preprocess import (
    val_preprocess,
    read_file_list,
    prepare_dataset_cls_tokens,
)
from utils.test_mIoU import mean_iou

def get_voc_palette(n=256):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='config/voc_test_ori_cfg.yaml', type=str)
    parser.add_argument('--load', dest='load_path', default='', type=str, help='Checkpoint path')
    parser.add_argument('--model', dest='model_name', default='SACR', type=str)
    return parser.parse_args()


def get_canny_edge(img_bytes, target_size=(224, 224)):

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, target_size)
    edges = cv2.Canny(img_resized, 100, 200)
    edge_tensor = torch.from_numpy(edges).float() / 255.0
    edge_tensor = edge_tensor.unsqueeze(0)
    return edge_tensor

def smart_load_state_dict(model, ckpt_path, device):
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and 'state_dict' in state:
        state_dict = state['state_dict']
    elif isinstance(state, dict):
        state_dict = state
    else:
        state_dict = state.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys (usually OK for inference): {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
    print("[INFO] Model weights loaded successfully.")


def main():
    args = get_parser()
    cfg = cfg_from_file(args.cfg_file)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(os.path.dirname(cfg.SAVE_DIR), f"test_run_{timestamp}")
    result_dir = os.path.join(base_dir, "results")
    vis_dir = os.path.join(base_dir, "visualization")
    cfg.SAVE_DIR = result_dir + '/'

    os.makedirs(result_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    print(f"[INFO] Results will be saved to: {base_dir}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    clip_model, _ = clip.load("ViT-B/16", device=device)

    model = SACR(cfg=cfg, clip_model=clip_model, rank=device).to(device)
    model.eval()

    _, _, _, _, val_images, val_labels, results_iou, _ = read_file_list(cfg)

    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT, map_location='cpu')
    text_weight = F.normalize(text_weight.float(), dim=1).to(device)
    cls_name_token, _ = prepare_dataset_cls_tokens(cfg)
    cls_name_token = cls_name_token.to(device)

    voc_palette = get_voc_palette(256)

    print("[INFO] Start Inference...")
    with torch.no_grad():
        for idx in tqdm(range(len(val_images)), desc="Testing"):
            with open(val_images[idx], 'rb') as f:
                value_buf = f.read()
            img_tensor = val_preprocess(cfg, value_buf).unsqueeze(0).to(device)
            edge_tensor = get_canny_edge(value_buf).to(device)
            img_metas = [{'edge_map': edge_tensor}]

            label_img = Image.open(val_labels[idx])
            w, h = label_img.size
            output = model(img_tensor, gt_cls=[], zeroshot_weights=text_weight,
                           cls_name_token=cls_name_token, training=False,
                           img_metas=img_metas)

            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            file_name = os.path.basename(val_images[idx]).split('.')[0]
            save_pt_path = os.path.join(result_dir, file_name + '.pt')
            torch.save(torch.from_numpy(pred).long(), save_pt_path)

            if args.save_vis:
                pred_img = Image.fromarray(pred)
                pred_img.putpalette(voc_palette)
                save_png_path = os.path.join(vis_dir, file_name + '.png')
                pred_img.save(save_png_path)

    print("[INFO] Calculating mIoU...")
    actual_results = [os.path.join(result_dir, os.path.basename(f).split('.')[0] + '.pt') for f in val_images]

    iou = mean_iou(actual_results, val_labels, num_classes=cfg.DATASET.NUM_CLASSES,
                   ignore_index=255, nan_to_num=0, reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL)
    print(f"\n[RESULT] mIoU: {np.nanmean(iou['IoU']) * 100:.2f}")


if __name__ == '__main__':
    main()