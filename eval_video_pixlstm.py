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
from lib.model.pixel_map import PIXLSTM_Residual
from lib.loss.mseloss import *
from lib.utils.img_hlp import *
from lib.utils.log_hlp import *
from dirs import *

DEF_MOD_RATE = 0.3
DEF_WAVE_FUNC = lambda x: np.cos(2*x*np.pi)*DEF_MOD_RATE+1-DEF_MOD_RATE
DEF_FLUSH_COUNT = -1
DEF_LSTM_STATE_SIZE=(322,322)

@torch.no_grad()
def myeval(model,skiprate:int=30,writer:bool=True):
    image_size=(640, 640)
    time_start = datetime.now()
    time_cur = time_start
    work_dir = os.path.join(DEF_WORK_DIR,'log','eva_{}_on_Minetto'.format(model.__class__.__name__))
    logger = SummaryWriter(os.path.join(work_dir,time_start.strftime("%Y%m%d-%H%M%S"))) if(writer)else None

    model = model.cuda() 
    criterion_mask = MSE_2d_Loss(pixel_sum=False).cuda()

    eval_dataset = Minetto(DEF_MINE_DIR,exclude_boxid=True)
    
    all_recall = []
    all_precision = []
    all_fscore = []
    all_loss = []
    for stepi, sample in enumerate(eval_dataset):
        vdo = sample['video']
        frame_bx_dict = sample['gt']
        p_keys = list(frame_bx_dict.keys())
        p_keys.sort()

        model.lstmc.zero_()
        model.lstmh.zero_()
        recall_list,precision_list,fscore_list,loss_list = [],[],[],[]
        fm_cnt,proc_cnt = 0,0
        # try:
        if(1):
            while(vdo.isOpened()):
                ret, x = vdo.read()
                if(ret==False):
                    break
                if(fm_cnt<p_keys[0] or fm_cnt%skiprate or fm_cnt not in p_keys):
                    # skip the initial non text frams
                    fm_cnt+=1
                    continue
                fgbxs = frame_bx_dict[fm_cnt]
                bgbxs = None

                fgbxs = np_box_resize(fgbxs,x.shape[-3:-1],image_size,'polyxy')
                x = cv2.resize(x,image_size[::-1])

                xnor = torch.from_numpy(np.expand_dims(x,0)).float().cuda()
                xnor = torch_img_normalize(xnor).permute(0,3,1,2)
                pred,feat = model(xnor)
                proc_cnt+=1
                region_np = pred[0,0].cpu().detach().numpy()
                region_np[region_np<0.2]=0.0

                det_boxes, label_mask, label_list = cv_get_box_from_mask(region_np,region_mean_split=True)
                if(det_boxes.shape[0]>0):
                    det_boxes = np_box_resize(det_boxes,region_np.shape[-2:],x.shape[-3:-1],'polyxy')
                    ids,mask_precision,mask_recall = cv_box_match(det_boxes,fgbxs,bgbxs,ovth=0.5)
                else:
                    mask_precision,mask_recall=0.0,0.0
                mask_fscore = calculate_fscore(mask_precision,mask_recall)
                recall_list.append(mask_recall)
                precision_list.append(mask_precision)
                fscore_list.append(mask_fscore)

                region_mask_np = cv_gen_gaussian_by_poly(fgbxs,image_size)
                region_mask = torch.from_numpy(region_mask_np.reshape(1,1,region_mask_np.shape[0],region_mask_np.shape[1])).float().cuda()
                loss = criterion_mask(pred[:,0:1],region_mask)
                loss_list.append(loss.item())

                if(logger):
                    region_np = pred[0,0].detach().cpu().numpy()
                    logger.add_image('X|GT|Pred in {}'.format(sample['name']),concatenate_images([cv_draw_poly(x,fgbxs),cv_heatmap(region_mask_np),cv_heatmap(region_np)]),fm_cnt,dataformats='HWC')
                    logger.add_scalar('Loss/ {}'.format(sample['name']),loss.item(),fm_cnt)
                    logger.add_scalar('Precision/ {}'.format(sample['name']),mask_precision,fm_cnt)
                    logger.add_scalar('Recall/ {}'.format(sample['name']),mask_recall,fm_cnt)
                    logger.add_scalar('F-score/ {}'.format(sample['name']),mask_fscore,fm_cnt)
                    logger.flush()
                fm_cnt += 1

        # except Exception as e:
        #     sys.stdout.write("Err at {}, frame {}, frame processed {}.\nErr: {}\n".format(sample['name'],fm_cnt,proc_cnt+1,str(e)))
        #     sys.stdout.flush()
        
        cur_recall=np.array(recall_list)
        cur_precision=np.array(precision_list)
        cur_fscore = np.array(fscore_list)
        cur_loss = np.array(loss_list)

        all_recall.append(cur_recall)
        all_precision.append(cur_precision)
        all_fscore.append(cur_fscore)
        all_loss.append(cur_loss)

        time_usage = datetime.now() - time_cur
        time_cur = datetime.now()
        sys.stdout.write("Video {}, Process {}/{}, Processed frame {}.\n".format(sample['name'],stepi+1,len(eval_dataset),proc_cnt))
        try:
            sys.stdout.write("Time usage: {} Second, {:5.2f} s/frame.\n".format(time_usage.seconds,time_usage.seconds/proc_cnt))
        except:
            sys.stdout.write("Time usage: {} Second, {:5.2f} s/frame.\n".format(time_usage.second,time_usage.second/proc_cnt))
        
        sys.stdout.write("\t Recall: Mean {:3.3f}%, variance {:3.3f}.\n".format(np.mean(cur_recall)*100,np.var(cur_recall)))
        sys.stdout.write("\t Precision: Mean {:3.3f}%, variance {:3.3f}.\n".format(np.mean(cur_precision)*100,np.var(cur_precision)))
        sys.stdout.write("\t F-score: Mean {:3.3f}%, variance {:3.3f}.\n".format(np.mean(cur_fscore)*100,np.var(cur_fscore)))
        sys.stdout.write("\t Loss: Mean {:2.5f}%, variance {:3.3f}.\n\n".format(np.mean(cur_loss),np.var(cur_loss)))
        sys.stdout.flush()

    recall_sum = 0.0
    for o in all_recall:
        recall_sum+=np.mean(o)
    recall_sum/=len(all_recall)

    precision_sum = 0.0
    for o in all_precision:
        precision_sum+=np.mean(o)
    precision_sum/=len(all_precision)

    final_fscore = calculate_fscore(recall_sum,precision_sum)
    sys.stdout.write("Final Recall: {:3.3f}%, Precision: {:3.3f}%, F-score: {:3.3f}%.\n".format(recall_sum*100,precision_sum*100,final_fscore*100))
    sys.stdout.flush()

    return 0

if __name__ == '__main__':
    pthdir = 'saved_model/lstm_ttt_Residual_region_pxnet.pth'
    model = PIXLSTM_Residual(mask_ch=2,basenet='mobile',min_upc_ch=128,min_map_ch=32,
        include_final=False,pretrained=True).float()
    model.init_state(shape=DEF_LSTM_STATE_SIZE,batch_size=1)
    model.load_state_dict(copyStateDict(torch.load(pthdir)))

    myeval(model)
