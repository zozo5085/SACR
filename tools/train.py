import argparse
import os
import ssl
import time
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

import clip
sys.path.append(os.getcwd())
ssl._create_default_https_context = ssl._create_unverified_context

from config.configs import cfg_from_file
from model.model import SACR
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, preprocess, read_file_list, prepare_dataset_cls_tokens


def custom_collate_fn(batch):
    imgs, labels, metas, filenames, pseudo_classes = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    return imgs, labels, metas, filenames, pseudo_classes


def total_variation_loss(x):
    b, c, h, w = x.size()
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2).sum()
    return (h_tv + w_tv) / (b * c * h * w)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', dest='cfg_file', default='config/voc_train_ori_cfg.yaml', type=str)
    p.add_argument('--val-interval', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--amp', action='store_true', help='use AMP')
    return p.parse_args()


class Train(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_filenames, _, self.train_images, self.train_labels, _, _, _, self.pseudo_classes = read_file_list(
            cfg)
        num_imgs = len(self.train_images)
        num_pseudo = len(self.pseudo_classes)

        if num_imgs != num_pseudo:
            raise RuntimeError(
                f"Pseudo labels count mismatch! images={num_imgs}, pseudo={num_pseudo}\n"
                f"First image: {self.train_images[0]}\n"
                f"Last  image: {self.train_images[-1]}"
            )

    def __getitem__(self, idx):
        with open(self.train_images[idx], 'rb') as f:
            value_buf = f.read()
        with open(self.train_labels[idx], 'rb') as f:
            label_buf = f.read()

        img, label, img_metas = preprocess(self.cfg, value_buf, label_buf, return_meta=True, unlabeled=False)
        return img, label, img_metas, self.train_images[idx], self.pseudo_classes[idx]

    def __len__(self):
        return len(self.train_images)


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power=0.9):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def pretty_print_iou(iou_vec, class_names, c_num, log_fp=None):
    vec = iou_vec[:c_num]
    lines = ["===== Per-Class IoU ====="]
    for i in range(c_num):
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        lines.append(f"{name}: {float(vec[i]):.4f}")
    miou = float(np.sum(vec) / c_num)
    lines.append("-------------------------")
    lines.append(f"mIoU: {miou:.4f}")
    text = "\n".join(lines)
    print(text)
    if log_fp is not None:
        print(text, file=log_fp)
        log_fp.flush()
    return miou


@torch.no_grad()
def validate_one_epoch(cfg, model, device, val_images, val_labels, val_filenames,
                       text_weight, cls_name_token, c_num, pred_save_dir, log_fp, class_names):
    model.eval()
    os.makedirs(pred_save_dir, exist_ok=True)

    clean_filenames = []
    for fn in val_filenames:
        fn = fn.strip()
        if ' ' in fn:
            fn = fn.split(' ')[0]
        while fn.startswith('/') or fn.startswith('\\'):
            fn = fn[1:]
        clean_filenames.append(fn)

    val_filenames = clean_filenames
    results_iou = [os.path.join(pred_save_dir, fn + ".pt") for fn in val_filenames]

    for idx in tqdm(range(len(val_images)), desc="Validating", leave=False):
        with open(val_images[idx], 'rb') as f:
            value_buf = f.read()
        img = val_preprocess(cfg, value_buf).unsqueeze(dim=0).to(device, non_blocking=True)

        label_img = Image.open(val_labels[idx])
        ori_shape = (label_img.size[1], label_img.size[0])

        output = model(img, gt_cls=[[]], zeroshot_weights=text_weight, cls_name_token=cls_name_token, training=False)
        output = F.interpolate(output, ori_shape, mode='bilinear', align_corners=False)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0).cpu()

        save_path = results_iou[idx]
        parent_dir = os.path.dirname(save_path)
        if not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except Exception as e:
                save_path = os.path.join(pred_save_dir, os.path.basename(clean_filenames[idx]) + '.pt')
                results_iou[idx] = save_path

        torch.save(pred, save_path)

    iou = mean_iou(
        results_iou, val_labels,
        num_classes=c_num + 1,
        ignore_index=cfg.DATASET.IGNORE_INDEX,
        nan_to_num=0,
        reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL
    )
    return pretty_print_iou(iou['IoU'], class_names, c_num, log_fp=log_fp)


def main():
    args = get_parser()
    cfg = cfg_from_file(args.cfg_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base_exp_dir = getattr(cfg, "SAVE_DIR", "experiments")
    exp_name = getattr(cfg, "EXP_NAME", "universal_train")
    run_dir = os.path.join(base_exp_dir, exp_name, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    pred_dir = os.path.join(run_dir, "val_preds")

    log_path = os.path.join(run_dir, 'log_train.txt')
    log = open(log_path, mode='a', encoding='utf-8')

    print(f"[INFO] Config: {args.cfg_file}")
    print(f"[INFO] Saving results to: {run_dir}")

    clip_model, _ = clip.load("ViT-B/16", device=device)

    train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, results_iou, pseudo_classes = read_file_list(
        cfg)
    cls_name_token, classes = prepare_dataset_cls_tokens(cfg)

    if isinstance(classes, (list, tuple)):
        class_names = classes
    else:
        class_names = [str(i) for i in range(cfg.DATASET.NUM_CLASSES)]

    print(f"[INFO] Loading text weights from: {cfg.DATASET.TEXT_WEIGHT}")
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT, map_location='cpu').to(device)
    text_weight = text_weight / (text_weight.norm(dim=1, keepdim=True) + 1e-6)

    train_data = Train(cfg)
    train_loader = DataLoader(
        dataset=train_data, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, batch_size=cfg.TRAIN.BATCH_SIZE,
        collate_fn=custom_collate_fn, drop_last=False
    )

    model = SACR(cfg=cfg, clip_model=clip_model, rank=0).to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0005)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    max_epoch = cfg.TRAIN.MAX_EPOCH
    c_num = cfg.DATASET.NUM_CLASSES
    best_iou = -1.0

    should_reduce_label = getattr(cfg.DATASET, "REDUCE_ZERO_LABEL", False)
    print(f"[INFO] REDUCE_ZERO_LABEL: {should_reduce_label} (Checked from Config)")

    print("[INFO] Start Training...")

    for epoch in range(max_epoch):
        model.train()
        running_loss = 0.0
        lr = adjust_learning_rate_poly(optimizer, epoch, max_epoch, cfg.TRAIN.LR)

        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}/{max_epoch}")

        for it, (img, label, img_metas, filenames, pseudo_class) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)
            img = img.to(device, non_blocking=True)


            gt_cls = []
            B = img.shape[0]
            for i in range(B):

                raw_labels = [int(t.item()) if hasattr(t, "item") else int(t) for t in pseudo_class[i]]


                final_labels = raw_labels

                gt_cls.append(final_labels)

            if epoch == 0 and it == 0:
                print(f"\n[DEBUG CHECK] Filename: {filenames[0]}")
                print(f"[DEBUG CHECK] Raw Pseudo (JSON): {pseudo_class[0]}")
                print(f"[DEBUG CHECK] Model Input: {gt_cls[0]}")
                print(
                    f"[DEBUG CHECK] Strategy: Reduce={should_reduce_label} (But JSON shift logic disabled for safety)")

            with torch.amp.autocast('cuda', enabled=args.amp):
                output, cls_loss = model(img, gt_cls, text_weight, cls_name_token, training=True, img_metas=img_metas)
                probs = F.softmax(output, dim=1)
                tv_loss = total_variation_loss(probs)

                loss = cls_loss + 0.005 * tv_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            loop.set_postfix(loss=float(loss.item()), lr=lr)

        print(f"Epoch {epoch + 1} finished. LR: {lr:.6f}. Loss: {running_loss / (len(train_loader) + 1e-6):.4f}",
              file=log)

        if args.val_interval > 0 and ((epoch + 1) % args.val_interval == 0):
            miou = validate_one_epoch(
                cfg, model, device,
                val_images, val_labels, val_filenames,
                text_weight, cls_name_token, c_num,
                pred_dir, log, class_names
            )

            if miou > best_iou:
                best_iou = miou
                torch.save(model.state_dict(), os.path.join(run_dir, f"best_miou_{best_iou:.4f}.pth"))
                print(f"[INFO] Saved Best: {best_iou:.4f}", file=log)

    log.close()


if __name__ == '__main__':
    main()