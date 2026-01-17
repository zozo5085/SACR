import yaml
from easydict import EasyDict as edict

cfg = edict()

cfg.DATASET = edict()
cfg.DATASET.NAME = ''
cfg.DATASET.NUM_CLASSES = 0
cfg.DATASET.REDUCE_ZERO_LABEL = True
cfg.DATASET.DATAROOT = ''
cfg.DATASET.SCALE = []
cfg.DATASET.RATIO_RANGE = []
cfg.DATASET.CROP_SIZE = []
cfg.DATASET.CAT_MAX_RATIO = 0
cfg.DATASET.TEXT_WEIGHT = ''
cfg.DATASET.IMG_NORM_CFG = edict()
cfg.DATASET.IMG_NORM_CFG.MEAN = []
cfg.DATASET.IMG_NORM_CFG.STD = []
cfg.DATASET.IMG_NORM_CFG.RGB = True
cfg.DATASET.K = 0
cfg.DATASET.DISTILL_K = 0
cfg.DATASET.THRESHOLD = 0
cfg.DATASET.IGNORE_INDEX = 255
cfg.DATASET.PALETTE = []

cfg.MODEL = edict()
cfg.MODEL.FEATURE_EXTRACTOR = ''
cfg.MODEL.TEXT_CHANNEL = 0
cfg.MODEL.VISUAL_CHANNEL = 0
cfg.MODEL.TRAINING = False

cfg.MODEL.USE_SA = False
cfg.MODEL.SA_HEADS = 8
cfg.MODEL.SA_GAMMA = 0.0
cfg.MODEL.PEPROJ_KERNEL = 1
cfg.MODEL.PEPROJ_DILATION = 1
cfg.MODEL.DECODER_KERNEL = 5
cfg.MODEL.DECODER_DILATION = 1

cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.MAX_EPOCH = 50
cfg.TRAIN.EPOCH = 0
cfg.TRAIN.MAX_ITER = 0
cfg.TRAIN.LR = 0.0
cfg.TRAIN.LOG = ''

cfg.TEST = edict()
cfg.TEST.BATCH_SIZE = 0
cfg.TEST.PD = 0.0
cfg.TEST.ReCLIP_PD = 0.5

cfg.EVAL_METRIC = ''
cfg.SAVE_DIR = ''
cfg.NUM_WORKERS = 0
cfg.LOAD_PATH = ''
cfg.LOAD_DISTILL_PATH = ''

def _to_edict(obj):
    if isinstance(obj, dict):
        return edict({k: _to_edict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_edict(v) for v in obj]
    return obj


def merge_a_to_b(a, b):
    if not isinstance(a, (edict, dict)):
        return b
    a = _to_edict(a) if not isinstance(a, edict) else a
    for k, v in a.items():
        if k in b and isinstance(v, (edict, dict)) and isinstance(b[k], (edict, dict)):
            merge_a_to_b(v, b[k])
        else:
            b[k] = _to_edict(v)
    return b

def cfg_from_file(filename):
    encodings = ("utf-8", "utf-8-sig", "cp950", "big5")
    last_err = None
    for enc in encodings:
        try:
            with open(filename, "r", encoding=enc) as f:
                text = f.read()
            y = yaml.safe_load(text) or {}
            y = _to_edict(y)
            return merge_a_to_b(y, cfg)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    with open(filename, "rb") as f:
        raw = f.read()
    y = yaml.safe_load(raw.decode("utf-8", errors="ignore")) or {}
    y = _to_edict(y)
    if not y and last_err:
        raise last_err
    return merge_a_to_b(y, cfg)
