from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

# from config.train_default import cfg as tcfg
cfg = {}

__DEF_SIZE = 0
cfg['IMG_SIZE'] = [
  [1280, 720], [int(32*40), int(32*23)], 
  [int(1280/2), int(720/2)], 
  [640, 640]
  ][1]

cfg['STEP'] = [1000*5,100*5,50*5,1*5][1]
cfg['BATCH'] = 4
cfg['LOGSTP'] = 100

cfg['OPT'] = ['sgd'][0]
# learning rate
cfg['LR'] = 0.001
# momentum
cfg['MMT'] = 0.8
# learning rate decrease step, 0 to disable
cfg['LR_DEC_STP'] = 1000
# learning rate decrease rate: current LR*rt
cfg['LR_DEC_RT'] = 0.9


cfg['DATASET'] = ["ttt","ctw","svt"][0]
