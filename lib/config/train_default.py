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

# train step, -1 for all
cfg['STEP'] = [1000*5,100*10,100*5,50*5,1*5,-1][-1]
cfg['BATCH'] = [4][0]
# Step size for log
cfg['LOGSTP'] = 100
# Step size for save model, -1 to disable 
cfg['SAVESTP'] = [200,500,-1][-1]
# Optimizer
cfg['OPT'] = ['sgd','adam'][1]
# learning rate
cfg['LR'] = [0.001,3.2768e-5][0]
# Momentum for sgd (..., momentum = cfg['MMT'])
cfg['MMT'] = 0.8
# For optimizer (...,weight_decay = cfg['OPT_DEC'])
cfg['OPT_DEC'] = 5e-4
# learning rate decrease step, 0 to disable
cfg['LR_DEC_STP'] = [0,500][0]
# learning rate decrease rate: current LR*(1-rt)
cfg['LR_DEC_RT'] = [0.2,0.1][1]

cfg['NET'] = ['craft', 'craft_mob'][0]

cfg['DATASET'] = ["ttt","ctw","svt",'sync','ic15'][-1]
