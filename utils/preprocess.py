import numpy as np
import cv2
from cv2 import IMREAD_COLOR
import torch
import os
import clip
from PIL import Image
import io
import random as py_random
import json
import re


prompt_templates = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.',
    'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
    'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.',
    'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
    'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.',
    'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.',
    'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
    'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.',
    'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.',
    'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
    'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
    'this is the {} in the scene.', 'this is one {} in the scene.',
]
voc_classes = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
               'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']
pascal_context_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'table', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    'bag', 'bed', 'bench', 'book', 'building', 'cabinet', 'ceiling', 'cloth', 'computer', 'cup',
    'curtain', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'keyboard', 'light',
    'mountain', 'mouse', 'plate', 'platform', 'road', 'rock', 'shelves', 'sidewalk', 'sign', 'sky',
    'snow', 'bedclothes', 'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood'
]
ade_classes = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet',
               'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
               'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence',
               'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
               'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator',
               'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway',
               'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench',
               'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
               'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning',
               'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land',
               'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
               'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket',
               'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name',
               'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood',
               'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor',
               'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']
coco_stuff_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                      'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                      'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                      'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                      'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                      'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building',
                      'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling', 'tile ceiling', 'cloth', 'clothes',
                      'clouds', 'counter', 'cupboard', 'curtain', 'desk', 'dirt', 'door', 'fence', 'marble floor',
                      'floor', 'stone floor', 'tile floor', 'wood floor', 'flower', 'fog', 'food', 'fruit', 'furniture',
                      'grass', 'gravel', 'ground', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror', 'moss',
                      'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant', 'plastic', 'platform',
                      'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand',
                      'sea', 'shelf', 'sky', 'skyscraper', 'snow', 'solid', 'stairs', 'stone', 'straw', 'structural',
                      'table', 'tent', 'textile', 'towel', 'tree', 'vegetable', 'brick wall', 'concrete wall', 'wall',
                      'panel wall', 'stone wall', 'tile wall', 'wood wall', 'water', 'waterdrops', 'blind window',
                      'window', 'wood']
coco_stuff_classes_27 = ["electronic", "appliance", "food things", "furniture things", "indoor", "kitchen", "accessory",
                         "animal", "outdoor", "person", "sports", "vehicle", "ceiling", "floor", "food stuff",
                         "furniture stuff", "raw material", "textile", "wall", "window", "building", "ground", "plant",
                         "sky", "solid", "structural", "water"]
cityscapes_classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                      'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                      'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def get_crop_bbox(cfg, img):
    crop_size = cfg.DATASET.CROP_SIZE

    margin_h = max(img.shape[0] - crop_size[0], 0)
    margin_w = max(img.shape[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
    return crop_y1, crop_y2, crop_x1, crop_x2


def crop(img, crop_bbox):
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
    return img


def preprocess(cfg, image, label, return_meta=False, unlabeled=True):
    img_metas = {}
    ratio_range = cfg.DATASET.RATIO_RANGE
    img_scale = cfg.DATASET.SCALE
    cat_max_ratio = cfg.DATASET.CAT_MAX_RATIO
    crop_size = cfg.DATASET.CROP_SIZE
    brightness_delta = 32
    contrast_range = (0.5, 1.5)
    contrast_lower, contrast_upper = contrast_range
    saturation_range = (0.5, 1.5)
    saturation_lower, saturation_upper = saturation_range
    hue_delta = 18

    img_metas['crop_size'] = crop_size
    img_np = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(img_np, IMREAD_COLOR)

    if label is not None:
        with io.BytesIO(label) as buff:
            label = Image.open(buff)
            label = np.array(label)
        label = label.astype(np.int64)
        if cfg.DATASET.REDUCE_ZERO_LABEL:
            label[label == 0] = 255
            label = label - 1
            label[label == 254] = 255

    h, w = img.shape[:2]
    min_ratio, max_ratio = ratio_range
    ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
    scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)

    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w),
                       max_short_edge / min(h, w))
    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    resized_img = cv2.resize(img, new_size, dst=None, interpolation=cv2.INTER_LINEAR)

    new_h, new_w = resized_img.shape[:2]
    w_scale = new_w / w
    h_scale = new_h / h
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                            dtype=np.float32)
    resized_label = cv2.resize(label, new_size, dst=None, interpolation=cv2.INTER_NEAREST)
    img_metas['new_size'] = new_size

    crop_bbox = get_crop_bbox(cfg, resized_img)
    for _ in range(10):
        seg_temp = crop(resized_label, crop_bbox)
        labels, cnt = np.unique(seg_temp, return_counts=True)
        cnt = cnt[labels != cfg.DATASET.IGNORE_INDEX]
        if len(cnt) > 1 and np.max(cnt) / np.sum(
                cnt) < cat_max_ratio:
            break
        crop_bbox = get_crop_bbox(cfg, resized_img)
    resized_img = crop(resized_img, crop_bbox)
    resized_label = crop(resized_label, crop_bbox)
    img_metas['crop_bbox'] = crop_bbox

    flip = True if np.random.rand() < 0.5 else False
    if flip:
        resized_img = np.flip(resized_img, axis=1)
        resized_img = resized_img.copy()
        resized_label = np.flip(resized_label, axis=1)
        resized_label = resized_label.copy()

    img = resized_img

    if py_random.randint(0, 2):
        beta = py_random.uniform(-brightness_delta, brightness_delta)
        img = img.astype(np.float32) + beta
        img = np.clip(img, 0, 255).astype(np.uint8)
    mode = py_random.randint(0, 2)
    if mode == 1:
        if py_random.randint(0, 2):
            alpha = py_random.uniform(contrast_lower, contrast_upper)
            img = img.astype(np.float32) * alpha
            img = np.clip(img, 0, 255).astype(np.uint8)

    if py_random.randint(0, 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        alpha = py_random.uniform(saturation_lower, saturation_upper)
        img[:, :, 1] = img[:, :, 1].astype(np.float32) * alpha
        img[:, :, 1] = np.clip(img[:, :, 1], 0, 255)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    if py_random.randint(0, 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 0] = (img[:, :, 0].astype(int) + py_random.randint(-hue_delta, hue_delta)) % 180
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    if mode == 0:
        if py_random.randint(0, 2):
            alpha = py_random.uniform(contrast_lower, contrast_upper)
            img = img.astype(np.float32) * alpha
            img = np.clip(img, 0, 255).astype(np.uint8)

    img = img.copy().astype(np.float32)
    mean = np.array(cfg.DATASET.IMG_NORM_CFG.MEAN, dtype=np.float32)
    std = np.array(cfg.DATASET.IMG_NORM_CFG.STD, dtype=np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    img = img.copy()
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)

    width = max(crop_size[1] - img.shape[1], 0)
    height = max(crop_size[0] - img.shape[0], 0)
    padding = (0, 0, width, height)
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        cv2.BORDER_CONSTANT,
        value=0)
    label = cv2.copyMakeBorder(
        resized_label,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        cv2.BORDER_CONSTANT,
        value=255)

    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    img = torch.from_numpy(img)
    label = torch.from_numpy(label.astype(np.int64))

    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray, 50, 150)

    edge_map = cv2.copyMakeBorder(
        edge_map,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        cv2.BORDER_CONSTANT,
        value=0
    )

    edge_map = edge_map.astype(np.float32) / 255.0
    edge_map = np.expand_dims(edge_map, axis=0)
    edge_map = torch.from_numpy(edge_map)

    img_metas['edge_map'] = edge_map

    if return_meta:
        return img, label, img_metas
    else:
        return img, label


def val_preprocess(cfg, image, label=None, unlabeled=True, distill=False):
    img_scale = cfg.DATASET.SCALE

    img_np = np.frombuffer(image, np.uint8)

    img = cv2.imdecode(img_np, IMREAD_COLOR)

    h, w = img.shape[:2]

    max_long_edge = max(img_scale)
    max_short_edge = min(img_scale)
    scale_factor = min(max_long_edge / max(h, w),
                       max_short_edge / min(h, w))

    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)

    resized_img = cv2.resize(
        img, new_size, dst=None, interpolation=cv2.INTER_LINEAR)

    new_h, new_w = resized_img.shape[:2]

    w_scale = new_w / w
    h_scale = new_h / h

    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                            dtype=np.float32)
    if label is not None:
        with io.BytesIO(label) as buff:
            label = Image.open(buff)
            label = np.array(label)
        label = label.astype(np.int64)
        if cfg.DATASET.REDUCE_ZERO_LABEL:
            label[label == 0] = 255
            label = label - 1
            label[label == 254] = 255
        if unlabeled:
            for i in range(cfg.DATASET.NUM_CLASSES):
                label[label == i] = -1
        label = cv2.resize(label, new_size, dst=None, interpolation=cv2.INTER_NEAREST)

    img = resized_img.copy().astype(np.float32)
    mean = np.array(cfg.DATASET.IMG_NORM_CFG.MEAN, dtype=np.float32)
    std = np.array(cfg.DATASET.IMG_NORM_CFG.STD, dtype=np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    img = img.copy()
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)

    img = torch.from_numpy(img.transpose(2, 0, 1))

    if distill:
        label = torch.from_numpy(label.astype(np.int64))
        return img, label

    return img

def load_pseudo_label_lines(txt_path):
    pseudo = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            s = s.strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1].strip()
            if s == "":
                pseudo.append([])
                continue
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            try:
                pseudo.append(sorted([int(p) for p in parts]))
            except ValueError:
                pseudo.append([])
    return pseudo

def read_file_list(cfg):
    dataset = cfg.DATASET.NAME
    if dataset == 'context':
        img_dir = os.path.join(cfg.DATASET.DATAROOT, 'JPEGImages')
        label_dir = os.path.join(cfg.DATASET.DATAROOT, 'SegmentationClassContext')
        set_dir = os.path.join(cfg.DATASET.DATAROOT, 'ImageSets', 'SegmentationContext')

        def read_txt_list(txt_name):
            txt_path = os.path.join(set_dir, txt_name)
            with open(txt_path, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]

        train_filenames = read_txt_list('train.txt')
        val_filenames = read_txt_list('val.txt')

        json_file_path = 'text/context_pseudo_label_SACR.json'
        if os.path.exists(json_file_path):
            pseudo_classes = load_pseudo_label_lines(json_file_path)
        else:
            pseudo_classes = []

        if len(pseudo_classes) > 0 and len(pseudo_classes) != len(train_filenames):
            min_len = min(len(train_filenames), len(pseudo_classes))
            train_filenames = train_filenames[:min_len]
            pseudo_classes = pseudo_classes[:min_len]

        train_images = [os.path.join(img_dir, f + '.jpg') for f in train_filenames]
        train_labels = [os.path.join(label_dir, f + '.png') for f in train_filenames]

        val_images = [os.path.join(img_dir, f + '.jpg') for f in val_filenames]
        val_labels = [os.path.join(label_dir, f + '.png') for f in val_filenames]

        output = [os.path.join(cfg.SAVE_DIR, f + '.pt') for f in val_filenames]

        if len(pseudo_classes) == 0:
            pseudo_classes = [[] for _ in range(len(train_images))]

        return (train_filenames, val_filenames,
                train_images, train_labels,
                val_images, val_labels,
                output, pseudo_classes)


    elif dataset == 'voc':
        train_txt_fname = os.path.join(cfg.DATASET.DATAROOT, 'ImageSets', 'Segmentation', 'train.txt')
        val_txt_fname = os.path.join(cfg.DATASET.DATAROOT, 'ImageSets', 'Segmentation', 'val.txt')

        with open(train_txt_fname, 'r') as f:
            train_filenames = f.read().split()
        with open(val_txt_fname, 'r') as f:
            val_filenames = f.read().split()

        train_images = [os.path.join(cfg.DATASET.DATAROOT, 'JPEGImages', i + '.jpg') for i in train_filenames]
        train_labels = [os.path.join(cfg.DATASET.DATAROOT, 'SegmentationClass', i + '.png') for i in train_filenames]
        val_images = [os.path.join(cfg.DATASET.DATAROOT, 'JPEGImages', i + '.jpg') for i in val_filenames]
        val_labels = [os.path.join(cfg.DATASET.DATAROOT, 'SegmentationClass', i + '.png') for i in val_filenames]
        output = [os.path.join(cfg.SAVE_DIR, i + '.pt') for i in val_filenames]

        pseudo_classes = []
        json_file_path = 'text/voc_pseudo_label_SACR.json'
        try:
            with open(json_file_path, 'r') as file:
                for line in file:
                    line = line[1:-2]
                    if ',' in line:
                        splitted_s = line.split(',')
                        splitted_s = list(map(int, splitted_s))
                        splitted_s.sort()
                    elif len(line) == 0:
                        splitted_s = []
                    else:
                        splitted_s = line
                        splitted_s = [int(splitted_s)]
                    pseudo_classes.append(splitted_s)
        except FileNotFoundError:
            pseudo_classes = [[] for _ in range(len(train_images))]

        return (train_filenames, val_filenames,
                train_images, train_labels,
                val_images, val_labels,
                output, pseudo_classes)

    elif dataset == 'ade':

        root = cfg.DATASET.DATAROOT.rstrip('/\\')
        val_img_dir = os.path.join(root, 'images', 'validation')
        train_img_dir = os.path.join(root, 'images', 'training')
        val_lbl_dir = os.path.join(root, 'annotations', 'validation')
        train_lbl_dir = os.path.join(root, 'annotations', 'training')

        val_filenames = sorted(
            [os.path.splitext(nm)[0] for nm in os.listdir(val_img_dir) if nm.lower().endswith('.jpg')])
        train_filenames = sorted(
            [os.path.splitext(nm)[0] for nm in os.listdir(train_img_dir) if nm.lower().endswith('.jpg')])

        val_images = [os.path.join(val_img_dir, f + '.jpg') for f in val_filenames]
        val_labels = [os.path.join(val_lbl_dir, f + '.png') for f in val_filenames]
        train_images = [os.path.join(train_img_dir, f + '.jpg') for f in train_filenames]
        train_labels = [os.path.join(train_lbl_dir, f + '.png') for f in train_filenames]

        output = [os.path.join(cfg.SAVE_DIR, f + '.pt') for f in val_filenames]

        pseudo_classes = []

        json_file_path = 'text/ade_pseudo_label_SACR.json'

        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as file:
                for raw in file:
                    line = raw.strip().strip("[] ")
                    if line == "":
                        splitted_s = []
                    else:
                        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                        try:
                            splitted_s = [int(p) for p in parts]
                            splitted_s.sort()
                        except ValueError:
                            splitted_s = []
                    pseudo_classes.append(splitted_s)
        else:
            pseudo_classes = [[] for _ in range(len(train_images))]

        return (train_filenames, val_filenames,
                train_images, train_labels,
                val_images, val_labels,
                output, pseudo_classes)


    elif dataset == 'stuff':

        root = cfg.DATASET.DATAROOT.rstrip('/\\')

        train_img_dir = os.path.join(root, 'images', 'train2017')

        val_img_dir = os.path.join(root, 'images', 'val2017')

        train_lbl_dir = os.path.join(root, 'annotations', 'train2017')

        val_lbl_dir = os.path.join(root, 'annotations', 'val2017')

        def list_stems(img_dir, sort_names=True):

            names = [nm for nm in os.listdir(img_dir) if nm.lower().endswith('.jpg')]

            if sort_names:
                names.sort()

            return [os.path.splitext(nm)[0] for nm in names]

        SORT_NAMES = True

        train_filenames = list_stems(train_img_dir, sort_names=SORT_NAMES)

        val_filenames = list_stems(val_img_dir, sort_names=SORT_NAMES)

        train_images = [os.path.join(train_img_dir, f + '.jpg') for f in train_filenames]

        train_labels = [os.path.join(train_lbl_dir, f + '_27labelTrainIds.png') for f in train_filenames]

        val_images = [os.path.join(val_img_dir, f + '.jpg') for f in val_filenames]

        val_labels = [os.path.join(val_lbl_dir, f + '_27labelTrainIds.png') for f in val_filenames]

        output = [os.path.join(cfg.SAVE_DIR, f + '.pt') for f in val_filenames]

        json_file_path = 'text/coco_pseudo_label_SACR.json'

        if os.path.exists(json_file_path):

            pseudo_classes = load_pseudo_label_lines(json_file_path)

        else:

            pseudo_classes = [[] for _ in range(len(train_images))]

        if len(pseudo_classes) != len(train_images):
            raise RuntimeError(
                f"COCO-stuff pseudo mismatch! train_images={len(train_images)}, pseudo={len(pseudo_classes)}"
            )

        return (train_filenames, val_filenames,

                train_images, train_labels,

                val_images, val_labels,

                output, pseudo_classes)


    elif dataset == 'cityscapes':

        train_txt_fname = os.path.join(cfg.DATASET.DATAROOT, 'train.txt')
        val_txt_fname = os.path.join(cfg.DATASET.DATAROOT, 'val.txt')

        with open(train_txt_fname, 'r') as f:
            train_filenames = f.read().strip().splitlines()

        with open(val_txt_fname, 'r') as f:
            val_filenames = f.read().strip().splitlines()

        def build_abs(base_dir, rel_path):
            return os.path.join(base_dir, rel_path.lstrip('/\\'))

        train_images = [build_abs(cfg.DATASET.DATAROOT, ln.split()[0]) for ln in train_filenames]
        train_labels = [build_abs(cfg.DATASET.DATAROOT, ln.split()[1]) for ln in train_filenames]
        val_images = [build_abs(cfg.DATASET.DATAROOT, ln.split()[0]) for ln in val_filenames]
        val_labels = [build_abs(cfg.DATASET.DATAROOT, ln.split()[1]) for ln in val_filenames]

        os.makedirs(cfg.SAVE_DIR, exist_ok=True)

        def stem_from_path(p):
            return os.path.splitext(os.path.basename(p))[0]

        output = [os.path.join(cfg.SAVE_DIR, f"{stem_from_path(p)}.pt") for p in val_images]

        pseudo_classes = []
        json_file_path = "text/cityscapes_pseudo_label_SACR.json"

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                raw = f.read().strip()
                try:
                    data = json.loads(raw)
                    if isinstance(data, list):
                        if all(isinstance(x, list) for x in data):
                            pseudo_classes = [sorted(int(v) for v in x) for x in data]
                        elif all(isinstance(x, dict) and "labels" in x for x in data):
                            pseudo_classes = [sorted(int(v) for v in x.get("labels", [])) for x in data]
                except json.JSONDecodeError:
                    for line in raw.splitlines():
                        line = line.strip()
                        if not line or line in ('[]', '[ ]'):
                            pseudo_classes.append([])
                        else:
                            line = line.strip('[]')
                            if not line.strip():
                                pseudo_classes.append([])
                            else:
                                vals = [s.strip() for s in line.split(',')]
                                vals = [int(s) for s in vals if s.isdigit()]
                                pseudo_classes.append(sorted(vals))
        except FileNotFoundError:
            pseudo_classes = [[] for _ in range(len(train_images))]

        if len(pseudo_classes) != len(train_images):
            if len(pseudo_classes) < len(train_images):
                pseudo_classes = pseudo_classes + [[]] * (len(train_images) - len(pseudo_classes))
            pseudo_classes = pseudo_classes[:len(train_images)]

        return (train_filenames, val_filenames,
                train_images, train_labels,
                val_images, val_labels,
                output, pseudo_classes)


def prepare_dataset_cls_tokens(cfg, noun=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = cfg.DATASET.NAME
    if dataset == 'context':
        dataset_classes = pascal_context_classes
    elif dataset == 'voc':
        dataset_classes = voc_classes
    elif dataset == 'ade':
        dataset_classes = ade_classes
    elif dataset == 'stuff':
        dataset_classes = coco_stuff_classes_27
    elif dataset == 'open':
        noun.append('background')
        dataset_classes = noun
        model, _ = clip.load("ViT-B/16")
        with torch.no_grad():
            zeroshot_weights = []
            for classname in noun:
                texts = [template.format(classname) for template in prompt_templates]
                texts = clip.tokenize(texts).cuda()
                class_embeddings = model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
            zeroshot_weights = zeroshot_weights.permute(1, 0).float()
    elif dataset == 'cityscapes' or dataset == 'gtav':
        dataset_classes = cityscapes_classes

    cls_name_token = []

    for cls_idx in range(len(dataset_classes)):
        cls_name = dataset_classes[cls_idx]
        token = clip.tokenize(cls_name)[0][1: 4]
        cls_name_token.append(token)
    cls_name_token = torch.stack(cls_name_token, dim=0)
    if dataset == 'open':
        return cls_name_token, zeroshot_weights.to(device)

    return cls_name_token, dataset_classes