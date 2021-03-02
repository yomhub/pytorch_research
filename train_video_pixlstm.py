import os
import sys
import platform
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
# =================Torch=======================
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# =================Local=======================
from lib.dataloader.icdar_video import ICDARV
from lib.dataloader.minetto import Minetto
from lib.model.pixel_map import PIXLSTM
from lib.loss.mseloss import *
from lib.utils.img_hlp import *
from lib.utils.log_hlp import *
from dirs import *

def train(args):
    image_size=(640, 640)
    time_start = datetime.now()
    work_dir = os.path.join(DEF_WORK_DIR,'log')
    if(args.name):
        work_dir = os.path.join(work_dir,args.name)
    logger = SummaryWriter(os.path.join(work_dir,time_start.strftime("%Y%m%d-%H%M%S"))) if(args.debug)else None
    
    model = PIXLSTM(mask_ch=2,min_map_ch=32,
        include_final=args.have_fc,basenet=args.basenet,min_upc_ch=128,pretrained=True).float()

    if(args.load and os.path.exists(args.load)):
        model.load_state_dict(copyStateDict(torch.load(args.load)))
    model = model.cuda()
    
    criterion = MSE_2d_Loss(pixel_sum=False).cuda(rank)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_gpu', help='Set --multi_gpu to enable multi gpu training.',action="store_true")
    parser.add_argument('--opt', help='PKL path or name of optimizer.',default='adag')
    parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
    parser.add_argument('--eval', help='Set --eval to enable eval.', action="store_true")
    parser.add_argument('--net', help='Choose noework.', default='PIX_Unet_MASK_CLS_BOX')
    parser.add_argument('--basenet', help='Choose base noework.', default='mobile')
    parser.add_argument('--tracker', type=str,help='Choose tracker.')
    parser.add_argument('--save', type=str, help='Set --save file_dir if want to save network.')
    parser.add_argument('--load', type=str, help='Set --load file_dir if want to load network.')
    parser.add_argument('--name', help='Name of task.')
    parser.add_argument('--dataset', help='Choose dataset: ctw/svt/ttt.', default='ttt')
    parser.add_argument('--batch', type=int, help='Batch size.',default=4)
    parser.add_argument('--learnrate', type=str, help='Learning rate.',default="0.001")
    parser.add_argument('--epoch', type=str, help='Epoch size.',default="10")
    parser.add_argument('--random', type=int, help='Set 1 to enable random change.',default=0)
    parser.add_argument('--bxrefine', help='Set --bxrefine to enable box refine.', action="store_true")
    parser.add_argument('--genmask', help='Set --genmask to enable generated mask.', action="store_true")
    parser.add_argument('--have_fc', help='Set --have_fc to include final level.', action="store_true")
    parser.add_argument('--lr_decay', help='Set --lr_decay to enbable learning rate decay.', action="store_true")

    args = parser.parse_args()
    args.dataset = args.dataset.lower() if(args.dataset)else args.dataset
    args.tracker = args.tracker.lower() if(args.tracker)else args.tracker

    # args.multi_gpu = False
    # args.debug = True
    # args.random=True
    # args.batch=2
    # args.load = "/BACKUP/yom_backup/saved_model/mask_only_pxnet.pth"
    # args.net = 'PIX_Unet_box'
    # args.dataset = 'msra'
    # args.eval=True

    gpus = torch.cuda.device_count()
    list_opt = args.opt.split(',')
    list_dataset = args.dataset.split(',')
    list_epoch = list(map(int,args.epoch.split(',')))
    list_learnrate = list(map(float,args.learnrate.split(',')))
    total_tasks = max(len(list_opt),len(list_dataset),len(list_epoch),len(list_learnrate))
    args_load = args.load
    last_save = None
    for taskid in range(total_tasks):
        if(taskid<len(list_opt)):
            cur_opt = list_opt[taskid]
        if(taskid<len(list_dataset)):
            cur_dataset = list_dataset[taskid]
        if(taskid<len(list_epoch)):
            cur_epoch = list_epoch[taskid]
        if(taskid<len(list_learnrate)):
            cur_learnrate = list_learnrate[taskid]
        if(not args_load and last_save):
            args.load = last_save
        summarize = "Start when {}.\n".format(datetime.now().strftime("%Y%m%d-%H%M%S")) +\
            "Task: {}/{}\n".format(taskid+1,total_tasks)+\
            "Working DIR: {}\n".format(DEF_WORK_DIR)+\
            "Running with: \n"+\
            "\t Epoch size: {},\n\t Batch size: {}.\n".format(args.epoch,args.batch)+\
            "\t Network: {}.\n".format(args.net)+\
            "\t Base network: {}.\n".format(args.basenet)+\
            "\t Include final level: {}.\n".format('Yes' if(args.have_fc)else 'No')+\
            "\t Optimizer: {}.\n".format(cur_opt)+\
            "\t LR decay: {}.\n".format('Yes' if(args.lr_decay)else 'No')+\
            "\t Dataset: {}.\n".format(cur_dataset)+\
            "\t Init learning rate: {}.\n".format(cur_learnrate)+\
            "\t Taks name: {}.\n".format(args.name if(args.name)else 'None')+\
            "\t Load network: {}.\n".format(args.load if(args.load)else 'No')+\
            "\t Save network: {}.\n".format(args.save if(args.save)else 'No')+\
            "\t Box refine: {}.\n".format('Yes' if(args.bxrefine)else 'No')+\
            "\t Generated mask: {}.\n".format('Yes' if(args.genmask)else 'No')+\
            "========\n"
            # "\t Train mask: {}.\n".format('Yes' if(DEF_BOOL_TRAIN_MASK)else 'No')+\
            # "\t Train classifier: {}.\n".format('Yes' if(DEF_BOOL_TRAIN_CE)else 'No')+\
        print(summarize)
        args.opt = cur_opt
        args.dataset = cur_dataset
        args.epoch = cur_epoch
        args.learnrate = cur_learnrate

        train(args)
        last_save = args.save