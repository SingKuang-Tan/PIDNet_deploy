
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

cfg = CN()

cfg.OUTPUT_DIR = ''
cfg.LOG_DIR = ''
cfg.GPUS = (0,)
cfg.WORKERS = 4
cfg.PRINT_FREQ = 20
cfg.AUTO_RESUME = False
cfg.PIN_MEMORY = True

# Cudnn related params
cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

# common params for NETWORK
cfg.MODEL = CN()
cfg.MODEL.NAME = 'pidnet_s'
cfg.MODEL.PRETRAINED = 'pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar'
cfg.MODEL.ALIGN_CORNERS = True
cfg.MODEL.NUM_OUTPUTS = 2
cfg.MODEL.SIZE = 'medium'

cfg.LOSS = CN()
cfg.LOSS.USE_OHEM = True
cfg.LOSS.USE_FocalTverskyLoss = False
cfg.LOSS.OHEMTHRES = 0.9
cfg.LOSS.BD_LABEL_THRES = 0.5
cfg.LOSS.OHEMKEEP = 100000
cfg.LOSS.CLASS_BALANCE = False
cfg.LOSS.BALANCE_WEIGHTS = [0.5, 0.5]
cfg.LOSS.SB_WEIGHTS = 0.5
cfg.LOSS.ALFA = 0.2

# DATASET related params
cfg.DATASET = CN()
cfg.DATASET.ROOT = 'data/'
cfg.DATASET.DATASET = 'cityscapes'
cfg.DATASET.NUM_CLASSES = 19
cfg.DATASET.TRAIN_FREIBURG_SET = 'list/cityscapes/train.lst'
cfg.DATASET.TRAIN_SET = 'list/cityscapes/train.lst'
cfg.DATASET.TEST_SET = 'list/cityscapes/test.lst'
cfg.DATASET.VALID_SET = 'list/cityscapes/val.lst'
cfg.DATASET.NONQUALIFIED_SET = ''
cfg.DATASET.EXTRA_TRAIN_SET = ''

# training
cfg.TRAIN = CN()
cfg.TRAIN.CLASS_WEIGHTS = []
cfg.TRAIN.TRUST_WEIGHTS = []
cfg.TRAIN.IMAGE_SIZE = [1024, 1024]  # width * height
cfg.TRAIN.MEAN = (0.4374, 0.4590, 0.4385)
cfg.TRAIN.STD = (0.1952, 0.2018, 0.2026)

cfg.TRAIN.LR = 0.01
cfg.TRAIN.EXTRA_LR = 0.001

cfg.TRAIN.OPTIMIZER = 'sgd'
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WD = 0.0001
cfg.TRAIN.NESTEROV = False
cfg.TRAIN.IGNORE_LABEL = -1
cfg.TRAIN.BEGIN_EPOCH = 0
cfg.TRAIN.END_EPOCH = 484
cfg.TRAIN.EXTRA_EPOCH = 0
cfg.TRAIN.RESUME_CHECKPOINT_PATH = ''
cfg.TRAIN.BATCH_SIZE_PER_GPU = 32
cfg.TRAIN.SHUFFLE = True

# validation
cfg.VALID = CN()
cfg.VALID.IMAGE_SIZE = [2048, 1024]  # width * height
cfg.VALID.BATCH_SIZE_PER_GPU = 32
cfg.VALID.MEAN = (0.4374, 0.4590, 0.4385)
cfg.VALID.STD = (0.1952, 0.2018, 0.2026)
cfg.VALID.MODEL_FILE = ''
cfg.VALID.OUTPUT_INDEX = -1

# testing
cfg.TEST = CN()
cfg.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
cfg.TEST.BATCH_SIZE_PER_GPU = 32
cfg.TEST.MEAN = (0.4374, 0.4590, 0.4385)
cfg.TEST.STD = (0.1952, 0.2018, 0.2026)
cfg.TEST.MODEL_FILE = ''
cfg.TEST.OUTPUT_INDEX = -1



def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg_file_path)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg 

