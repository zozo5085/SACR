from PIL import Image
import numpy as np
import torch
from collections import OrderedDict
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float64

def imread(path):
    img = Image.open(path)
    array = np.array(img)
    if array.ndim >= 3 and array.shape[2] >= 3:
        array[:, :, :3] = array[:, :, (2, 1, 0)]
    return array


def intersect_and_union(pred_path,
                        label_path,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):

    pred = torch.load(pred_path, map_location=DEVICE)
    if pred.ndim == 4 and pred.shape[1] > 1:
        pred = torch.argmax(pred, dim=1).squeeze(0)
    elif pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)
    pred = pred.to(DEVICE).long()

    lab_np = np.array(Image.open(label_path))
    lab = torch.from_numpy(lab_np).to(DEVICE).long()

    if label_map:
        for old_id, new_id in label_map.items():
            lab[lab == old_id] = new_id
    if reduce_zero_label:
        lab[lab == 0] = 255
        lab = lab - 1
        lab[lab == 254] = 255

    valid = (lab != ignore_index) & (lab >= 0) & (lab < num_classes)

    pred_v = pred[valid]
    lab_v  = lab[valid]

    inter = pred_v[pred_v == lab_v]

    area_intersect = torch.bincount(inter, minlength=num_classes).to(DEVICE, DTYPE)
    area_pred      = torch.bincount(pred_v, minlength=num_classes).to(DEVICE, DTYPE)
    area_label     = torch.bincount(lab_v,  minlength=num_classes).to(DEVICE, DTYPE)
    area_union     = area_pred + area_label - area_intersect

    return area_intersect, area_union, area_pred, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    total_area_intersect = torch.zeros((num_classes,), dtype=DTYPE, device=DEVICE)
    total_area_union     = torch.zeros((num_classes,), dtype=DTYPE, device=DEVICE)
    total_area_pred      = torch.zeros((num_classes,), dtype=DTYPE, device=DEVICE)
    total_area_label     = torch.zeros((num_classes,), dtype=DTYPE, device=DEVICE)

    for res, gt in tqdm(zip(results, gt_seg_maps),
                        total=len(results), desc='Evaluating IoU', ncols=80, leave=False):
        ai, au, ap, al = intersect_and_union(
            res, gt, num_classes, ignore_index, label_map, reduce_zero_label)

        total_area_intersect += ai
        total_area_union     += au
        total_area_pred      += ap
        total_area_label     += al

    return total_area_intersect, total_area_union, total_area_pred, total_area_label


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            precision = total_area_intersect / total_area_pred_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
            ret_metrics['Prec'] = precision
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc

    ret_metrics = {
        metric: value.cpu().numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
        results, gt_seg_maps, num_classes, ignore_index, label_map,
        reduce_zero_label)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result