import os
import sys
import torch
import argparse
from datetime import datetime
# =================Torch=======================
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# =================Local=======================
from lib.model.craft import CRAFT
from lib.dataloader.total import Total
from lib.dataloader.icdar_video import ICDARV
from lib.utils.img_hlp import RandomScale
from lib.config.train_default import cfg as tcfg


__DEF_LOCAL_DIR = os.path.split(__file__)[0]
__DEF_DATA_DIR = os.path.join(__DEF_LOCAL_DIR, 'mydataset')
__DEF_CTW_DIR = os.path.join(__DEF_DATA_DIR, 'ctw')
__DEF_SVT_DIR = os.path.join(__DEF_DATA_DIR, 'svt')
__DEF_TTT_DIR = os.path.join(__DEF_DATA_DIR, 'totaltext')
__DEF_ICV_DIR = os.path.join(__DEF_DATA_DIR, 'TextVideo')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config trainer')

    parser.add_argument('--opt', help='Choose optimizer.',default=tcfg['OPT'])
    parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
    parser.add_argument('--save', help='Set --save if want to save network.', action="store_true")
    parser.add_argument('--load', help='Set --load if want to load network.', action="store_true")
    parser.add_argument('--net', help='Choose noework (craft).', default="craft")
    parser.add_argument('--name', help='Name of task.')
    parser.add_argument('--dataset', help='Choose dataset: ctw/svt/ttt.', default=tcfg['DATASET'])
    parser.add_argument('--datax', type=int, help='Dataset output width.',default=tcfg['IMG_SIZE'][0])
    parser.add_argument('--datay', type=int, help='Dataset output height.',default=tcfg['IMG_SIZE'][1])
    parser.add_argument('--step', type=int, help='Step size.',default=tcfg['STEP'])
    parser.add_argument('--batch', type=int, help='Batch size.',default=tcfg['BATCH'])
    parser.add_argument('--logstp', type=int, help='Log step size.',default=tcfg['LOGSTP'])
    parser.add_argument('--gpu', type=int, help='Set --gpu -1 to disable gpu.',default=0)
    parser.add_argument('--savestep', type=int, help='Batch size.',default=20)
    parser.add_argument('--learnrate', type=float, help='Learning rate.',default=tcfg['LR'])

    args = parser.parse_args()
    time_start = datetime.now()
    isdebug = args.debug
    lr = args.learnrate

    summarize = "Start when {}.\n".format(time_start.strftime("%Y%m%d-%H%M%S")) +\
        "Running with: \n\t Use proposal: {},\n\t Is debug: {}.\n".format(args.proposal,args.debug)+\
        "\t Step size: {},\n\t Batch size: {}.\n".format(args.step,args.batch)+\
        "\t Input shape: x={},y={}.\n".format(args.datax,args.datay)+\
        "\t Optimizer: {}.\n".format(args.opt)+\
        "\t Init learning rate: {}.\n".format(lr)+\
        "\t Taks name: {}.\n".format(args.name)+\
        "\t Use GPU: {}.\n".format('Yes' if(args.gpu>=0)else 'No')+\
        "\t Save network: {}.\n".format('Yes' if(args.save)else 'No')+\
        "\t Load network: {}.\n".format('Yes' if(args.load)else 'No')
    print(summarize)

    net = CRAFT()
    opt = optim.SGD(net.parameters(), lr=lr, momentum=tcfg['MMT'])
    pass
