import numpy as np
from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 128

config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [8, 14]
