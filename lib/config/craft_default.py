from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

# from config.craft_default import cfg as tcfg
cfg = {}


cfg['EPOCH'] = 200
cfg['BATCH'] = 2

# optimizer
cfg['OPTIMIZER'] = ['sgd'][0]
# momentum
cfg['MMT'] = 0.8

# learning rate
cfg['LR_RATE'] = 0.001
# learning rate decrease step, 0 to disable
cfg['LR_DEC_STP'] = 200
# learning rate decrease rate: current LR*(1-rt)
cfg['LR_DEC_RT'] = 0.1