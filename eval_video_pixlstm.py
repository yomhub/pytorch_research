import os
import sys
import platform
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pickle
# =================Torch=======================
import torch
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# =================Local=======================
from lib.dataloader.icdar_video import ICDARV
from lib.dataloader.minetto import Minetto
from lib.model.pixel_map import PIXLSTM_Residual,PIX_Unet,PIXCNN
from lib.loss.mseloss import *
from lib.utils.img_hlp import *
from lib.utils.log_hlp import *
from dirs import *

DEF_MOD_RATE = 0.3
DEF_WAVE_FUNC = lambda x: np.cos(2*x*np.pi)*DEF_MOD_RATE+1-DEF_MOD_RATE
DEF_FLUSH_COUNT = -1
DEF_LSTM_STATE_SIZE=(322,322)

@torch.no_grad()
def myeval(model_list,maskch_list,model_name_list,skiprate:int=1,writer:bool=True):
    image_size=(640, 640)
    time_start = datetime.now()
    time_cur = time_start

    if(not model_name_list or len(model_name_list)<len(model_list)):
        if(not model_name_list):
            model_name_list=[]
        model_name_list += [model.__class__.__name__ for model in model_list[len(model_name_list):]]

    eval_dataset = Minetto(DEF_MINE_DIR)
    task_name = 'eva_{}_'.format(eval_dataset.__class__.__name__)
    for o in model_name_list:
        task_name+='_'+o
    
    work_dir = os.path.join(DEF_WORK_DIR,'log',task_name)

    model_list = [model.cuda() for model in model_list]
    all_eva_dict = {o:defaultdict(list) for o in model_name_list}
    criterion_mask = MSE_2d_Loss(pixel_sum=False).cuda()

    loggers = []
    if(writer):
        for o in model_name_list:
            loggers.append(SummaryWriter(os.path.join(work_dir,o)))
    else:
        loggers = [None]*len(model_name_list)

    for stepi, sample in enumerate(eval_dataset):
        eva_dict = {o:defaultdict(list) for o in model_name_list}
        vdo = sample['video']
        frame_bx_dict = sample['gt']
        p_keys = list(frame_bx_dict.keys())
        p_keys.sort()
        try:
            model.lstmc.zero_()
            model.lstmh.zero_()
        except:
            None

        fm_cnt,proc_cnt = 0,0

        while(vdo.isOpened()):
            ret, x = vdo.read()
            if(ret==False):
                break
            if(fm_cnt<p_keys[0] or fm_cnt%skiprate):
                # skip the initial non text frams
                fm_cnt+=1
                continue

            org_size = x.shape[-3:-1]
            x = cv2.resize(x,image_size[::-1])
            xnor = torch.from_numpy(np.expand_dims(x,0)).float().cuda()
            xnor = torch_img_normalize(xnor).permute(0,3,1,2)
            
            if(fm_cnt in frame_bx_dict):
                boxs = frame_bx_dict[fm_cnt]
                boxs = np_box_resize(boxs,org_size,image_size,'polyxy')
                txts = sample['txt'][fm_cnt]
                fgbxs = np.array([bx for bx,tx in zip(boxs,txts) if(txts!='#')])
                bgbxs = np.array([bx for bx,tx in zip(boxs,txts) if(txts=='#')])
            else:
                fgbxs = np.array([])
                bgbxs = np.array([])

            proc_cnt+=1
            for model,model_name,maskch,logger in zip(model_list,model_name_list,maskch_list,loggers):
                try:
                    model.lstmc.zero_()
                    model.lstmh.zero_()
                except:
                    None

                pred,feat = model(xnor)
                region_np = pred[0,maskch].cpu().detach().numpy()
                region_np[region_np<0.2]=0.0

                det_boxes, label_mask, label_list = cv_get_box_from_mask(region_np,region_mean_split=True)

                if(fgbxs.size==0):
                    mask_precision=1
                    mask_recall = 1 if(det_boxes.size==0)else 0
                elif(det_boxes.shape[0]>0):
                    det_boxes = np_box_resize(det_boxes,region_np.shape[-2:],x.shape[-3:-1],'polyxy')
                    ids,mask_precision,mask_recall = cv_box_match(det_boxes,fgbxs,bgbxs,ovth=0.5)
                else:
                    mask_precision,mask_recall=0.0,1
            
                mask_fscore = calculate_fscore(mask_precision,mask_recall)
                eva_dict[model_name]['recall'].append(mask_recall)
                eva_dict[model_name]['precision'].append(mask_precision)
                eva_dict[model_name]['fscore'].append(mask_fscore)
                eva_dict[model_name]['step'].append(fm_cnt)

                if(fgbxs.size>0):
                    region_mask_np = cv_gen_gaussian_by_poly(fgbxs,image_size)
                else:
                    region_mask_np = np.zeros(image_size,np.float32)
                    
                region_mask = torch.from_numpy(region_mask_np.reshape(1,1,region_mask_np.shape[0],region_mask_np.shape[1])).float().cuda()
                loss = criterion_mask(pred[:,maskch:maskch+1],region_mask)
                eva_dict[model_name]['loss'].append(loss.item())

                if(logger):
                    region_np = pred[0,maskch].detach().cpu().numpy()
                    img = x
                    img = cv_draw_poly(img,fgbxs,'GT',color=(255,0,0))
                    img = cv_draw_poly(img,det_boxes,'Pred',color=(0,255,0))

                    logger.add_image('X|GT|Pred in {}'.format(sample['name']),concatenate_images([img,cv_heatmap(region_mask_np),cv_heatmap(region_np)]),fm_cnt,dataformats='HWC')
                    logger.add_scalar('Loss/ {}'.format(sample['name']),loss.item(),fm_cnt)
                    logger.add_scalar('Precision/ {}'.format(sample['name']),mask_precision,fm_cnt)
                    logger.add_scalar('Recall/ {}'.format(sample['name']),mask_recall,fm_cnt)
                    logger.add_scalar('F-score/ {}'.format(sample['name']),mask_fscore,fm_cnt)
                    logger.flush()
            fm_cnt += 1

        sys.stdout.write("Video {}, Process {}/{}, Processed frame {}.\n".format(sample['name'],stepi+1,len(eval_dataset),proc_cnt))
        for model_name in model_name_list:
            
            cur_recall = np.array(eva_dict[model_name]['recall'])
            cur_precision = np.array(eva_dict[model_name]['precision'])
            cur_fscore = np.array(eva_dict[model_name]['fscore'])
            cur_loss = np.array(eva_dict[model_name]['loss'])
            cur_step = np.array(eva_dict[model_name]['step'])

            all_eva_dict[model_name]['recall'].append(cur_recall)
            all_eva_dict[model_name]['precision'].append(cur_precision)
            all_eva_dict[model_name]['fscore'].append(cur_fscore)
            all_eva_dict[model_name]['loss'].append(cur_loss)
            all_eva_dict[model_name]['step'].append(cur_step)

            sys.stdout.write(
                '======================' + \
                '\t Model: {}\n'.format(model_name) +\
                '\t Recall: Mean {:3.3f}%, variance {:3.3f}.\n'.format(np.mean(cur_recall)*100,np.var(cur_recall))+\
                '\t Precision: Mean {:3.3f}%, variance {:3.3f}.\n'.format(np.mean(cur_precision)*100,np.var(cur_precision))+\
                '\t F-score: Mean {:3.3f}%, variance {:3.3f}.\n'.format(np.mean(cur_fscore)*100,np.var(cur_fscore))+\
                '\t Loss: Mean {:2.5f}, variance {:3.3f}.\n'.format(np.mean(cur_loss),np.var(cur_loss))+\
                '\n'
            )
            sys.stdout.flush()
        # break


    for model_name in model_name_list:
        all_recall = all_eva_dict[model_name]['recall']
        recall_sum = 0.0
        for o in all_recall:
            recall_sum+=np.mean(o)
        recall_sum/=len(all_recall)

        all_precision = all_eva_dict[model_name]['precision']
        precision_sum = 0.0
        for o in all_precision:
            precision_sum+=np.mean(o)
        precision_sum/=len(all_precision)

        all_loss = all_eva_dict[model_name]['loss']
        loss_sum = 0.0
        for o in all_loss:
            loss_sum+=np.mean(o)
        loss_sum/=len(all_loss)

        final_fscore = calculate_fscore(recall_sum,precision_sum)
        sys.stdout.write("Final Recall: {:3.3f}%, Precision: {:3.3f}%, F-score: {:3.3f}%, Loss: {:4.3f}.\n".format(
            recall_sum*100,precision_sum*100,final_fscore*100,loss_sum))
        sys.stdout.flush()

    return all_eva_dict

if __name__ == '__main__':
    work_dir = '/home/yomcoding/Pytorch/MyResearch/saved_model/'

    model_list,maskch_list,model_name_list = [],[],[]

    pthdir = '/BACKUP/yom_backup/saved_model/lstm_ttt_Residual_region_pxnet.pth'
    model_lstm = PIXLSTM_Residual(mask_ch=2,basenet='mobile',min_upc_ch=128,min_map_ch=32,
        include_final=False,pretrained=True).float()
    model_lstm.init_state(shape=DEF_LSTM_STATE_SIZE,batch_size=1)
    model_lstm.load_state_dict(copyStateDict(torch.load(pthdir)))
    model_list.append(model_lstm)
    maskch_list.append(0)
    model_name_list.append('LSTM')

    # pthdir = '/BACKUP/yom_backup/saved_model/CNN_ttt_region_pxnet.pth'
    # model_cnn = PIXCNN(mask_ch=2,basenet='mobile',min_upc_ch=128,min_map_ch=32,
    #     include_final=False,pretrained=True).float()
    # model_cnn.load_state_dict(copyStateDict(torch.load(pthdir)))
    # model_list.append(model_cnn)
    # maskch_list.append(0)
    # model_name_list.append('CNN')

    # pthdir = '/BACKUP/yom_backup/saved_model/org_ttt_region_pxnet.pth'
    # model_cnn_random = PIX_Unet(mask_ch=4,min_mask_ch=32,include_final=False,basenet='mobile',min_upc_ch=128,pretrained=False).float()
    # model_cnn_random.load_state_dict(copyStateDict(torch.load(pthdir)))
    # model_list.append(model_cnn_random)
    # maskch_list.append(2)
    # model_name_list.append('CNN_random')

    all_eva_dict_k1 = myeval(model_list,maskch_list,model_name_list,skiprate=1)
    all_eva_dict_k10 = myeval(model_list,maskch_list,model_name_list,skiprate=10)
    all_eva_dict_k20 = myeval(model_list,maskch_list,model_name_list,skiprate=20)
    all_eva_dict_k30 = myeval(model_list,maskch_list,model_name_list,skiprate=30)
    kdicts = [all_eva_dict_k1,all_eva_dict_k10,all_eva_dict_k20,all_eva_dict_k30]
    ks = [1,10,20,30]
    # with open(os.path.join(work_dir,'all_eva_mintto.pkl'), 'wb') as pfile:
    #     pickle.dump(all_eva_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    
    total_nums = len(all_eva_dict_k1['LSTM']['loss'])
    all_val_list = [[] for _ in range(total_nums)]
    for all_eva_dict,ksize in zip(kdicts,ks):
        xs_list = all_eva_dict['LSTM']['step']
        loss_list = all_eva_dict['LSTM']['loss']
        for i,(xs,loss) in enumerate(zip(xs_list,loss_list)):
            all_val_list[i].append(
                (xs,loss,ksize)
            )

    for i,val_list in enumerate(all_val_list):
        fig,axs = plt.subplots(1,1)
        axs.get_yaxis().label.set_text('Loss')
        axs.get_xaxis().label.set_text('Frame Step')
        for xs,loss,ksize in val_list:
            axs.plot(xs,loss,label = 'Frame step {}'.format(ksize))
        axs.legend()
        fig.savefig(os.path.join(work_dir,'loss_mintto_{}.png'.format(i)))