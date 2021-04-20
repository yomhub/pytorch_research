import os
import sys
import platform
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
# =================Torch=======================
import torch
import torch.optim as optim
import cv2
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

DEF_MOD_RATE = 0.3
DEF_WAVE_FUNC = lambda x: np.cos(2*x*np.pi)*DEF_MOD_RATE+1-DEF_MOD_RATE
DEF_FLUSH_COUNT = -1
DEF_LSTM_STATE_SIZE=(322,322)

@torch.no_grad()
def eavl(model,skiprate:int=30):
    image_size=(640, 640)
    time_start = datetime.now()
    time_cur = time_start
    work_dir = os.path.join(DEF_WORK_DIR,'log')
    if(args.name):
        work_dir = os.path.join(work_dir,args.name)
    logger = SummaryWriter(os.path.join(work_dir,time_start.strftime("%Y%m%d-%H%M%S"))) if(not args.debug)else None
    log = sys.stdout

    model = model.float().cuda() 
    model.init_state(shape=DEF_LSTM_STATE_SIZE,batch_size=1)
    criterion_mask = MSE_2d_Loss(pixel_sum=False).cuda()

    eval_dataset = Minetto(DEF_MINE_DIR)
    batch_size = 1
    
    all_recall = []
    all_precision = []
    for stepi, sample in enumerate(eval_dataset):
        vdo = sample['video']
        frame_bx_dict = sample['gt']
        p_keys = list(frame_bx_dict.keys())
        p_keys.sort()

        model.lstmc.zero_()
        model.lstmh.zero_()
        recall_list,precision_list = [],[]
        fm_cnt,proc_cnt = 0,0
        try:
        # if(1):
            while(vdo.isOpened()):
                ret, x = vdo.read()
                if(ret==False):
                    break
                if(fm_cnt<p_keys[0] or fm_cnt%args.skiprate):
                    # skip the initial non text frams
                    fm_cnt+=1
                    continue
                fgbxs = frame_bx_dict[fm_cnt]
                bgbxs = None

                x = cv2.resize(x,image_size[::-1])
                xnor = torch.from_numpy(np.expand_dims(x,0)).float().cuda()
                xnor = torch_img_normalize(xnor).permute(0,3,1,2)
                pred,feat = model(xnor)
                proc_cnt+=1
                pred_np = pred[0,0].cpu().detach().numpy()
                pred_np[pred_np<0.2]=0.0

                det_boxes, label_mask, label_list = cv_get_box_from_mask(region_np,region_mean_split=True)
                if(det_boxes.shape[0]>0):
                    det_boxes = np_box_resize(det_boxes,region_np.shape[-2:],x.shape[-3:-1],'polyxy')
                    ids,mask_precision,mask_recall = cv_box_match(det_boxes,fgbxs,bgbxs,ovth=0.5)
                else:
                    mask_precision,mask_recall=0.0,0.0
                recall_list.append(mask_recall)
                precision_list.append(mask_precision)

                loss_dict = {}
                region_mask = torch.from_numpy(region_mask_np.reshape(1,1,region_mask_np.shape[0],region_mask_np.shape[1])).float().cuda()
                loss_dict['region_loss'] = criterion_mask(pred[:,0:1],region_mask)

                loss = 0.0
                for keyn,value in loss_dict.items():
                    loss+=value

                if(DEF_FLUSH_COUNT>0 and (proc_cnt+1)%DEF_FLUSH_COUNT==0):
                    model.init_state(shape=DEF_LSTM_STATE_SIZE,batch_size=1)

                if(logger):
                    region_np = pred[0,0].detach().cpu().numpy()
                    logger.add_image('Region: s{},e{}'.format(stepi,epoch),concatenate_images([x,cv_heatmap(region_np)]),fm_cnt,dataformats='HWC')
                fm_cnt += 1

        except Exception as e:
            sys.stdout.write("Err at {}, frame {}, frame processed {}.\nErr: {}\n".format(sample['name'],fm_cnt,proc_cnt+1,str(e)))
            sys.stdout.flush()
        
        cur_recall=np.array(recall_list)
        cur_precision=np.array(precision_list)
        mean_recall=np.mean(cur_recall)
        mean_precision=np.mean(precision_list)
        mean_fscore = 2*mean_recall*mean_precision/(mean_recall+mean_precision) if(mean_recall+mean_precision>0)else 0.0
        all_recall.append(cur_recall)
        all_precision.append(cur_precision)
    
        time_usage = datetime.now() - time_cur
        time_cur = datetime.now()
        sys.stdout.write("Video {}, Process {}/{}, Processed frame {}.\n".format(sample['name'],stepi+1,len(eval_dataset),proc_cnt))
        try:
            sys.stdout.write("Time usage: {} Second, {:5.2f} s/frame.\n".format(time_usage.seconds,proc_cnt/time_usage.seconds))
        except:
            sys.stdout.write("Time usage: {} Second, {:5.2f} s/frame.\n".format(time_usage.second,proc_cnt/time_usage.second))
        
        sys.stdout.write("\t Recall: Mean {:3.3f}%, variance {:3.3f}.\n".format(np.mean(cur_recall)*100,np.var(cur_recall)))
        sys.stdout.write("\t Precision: Mean {:3.3f}%, variance {:3.3f}.\n".format(np.mean(cur_precision)*100,np.var(cur_precision)))
        sys.stdout.write("\t F-score: Mean {:3.3f}%.\n\n".format(mean_fscore))
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_gpu', help='Set --multi_gpu to enable multi gpu training.',action="store_true")
    parser.add_argument('--opt', help='PKL path or name of optimizer.',default='adag')
    parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
    parser.add_argument('--eval', help='Set --eval to enable eval.', action="store_true")
    parser.add_argument('--basenet', help='Choose base noework.', default='mobile')
    parser.add_argument('--tracker', type=str,help='Choose tracker.')
    parser.add_argument('--save', type=str, help='Set --save file_dir if want to save network.')
    parser.add_argument('--load', type=str, help='Set --load file_dir if want to load network.')
    parser.add_argument('--name', help='Name of task.')
    parser.add_argument('--dataset', help='Choose dataset: Minetto/icv15.', default='Minetto')
    parser.add_argument('--learnrate', type=str, help='Learning rate.',default="0.001")
    parser.add_argument('--epoch', type=str, help='Epoch size.',default="10")
    parser.add_argument('--lr_decay', help='Set --lr_decay to enbable learning rate decay.', action="store_true")
    # LSTM specific
    parser.add_argument('--linear', help='Set --linear to enbable linearly box plugin.', action="store_true")
    parser.add_argument('--skiprate', type=int, help='Fram skip rate.',default=3)

    args = parser.parse_args()
    args.dataset = args.dataset.lower() if(args.dataset)else args.dataset
    args.tracker = args.tracker.lower() if(args.tracker)else args.tracker

    # args.debug = True

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
            "\t Epoch size: {}.\n".format(args.epoch)+\
            "\t Base network: {}.\n".format(args.basenet)+\
            "\t Optimizer: {}.\n".format(cur_opt)+\
            "\t LR decay: {}.\n".format('Yes' if(args.lr_decay)else 'No')+\
            "\t Dataset: {}.\n".format(cur_dataset)+\
            "\t Init learning rate: {}.\n".format(cur_learnrate)+\
            "\t Taks name: {}.\n".format(args.name if(args.name)else 'None')+\
            "\t Load network: {}.\n".format(args.load if(args.load)else 'No')+\
            "\t Save network: {}.\n".format(args.save if(args.save)else 'No')+\
            "\t Skip rate: {}.\n".format(args.skiprate)+\
            "\t State flush count: {}.\n".format(DEF_FLUSH_COUNT)+\
            "\t Linear: {}.\n".format('Yes' if(args.linear)else 'No')+\
            "========\n"
        print(summarize)

        args.opt = cur_opt
        args.dataset = cur_dataset
        args.epoch = cur_epoch
        args.learnrate = cur_learnrate

        train(args)
        last_save = args.save