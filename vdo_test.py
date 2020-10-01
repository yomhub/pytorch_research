import os
import sys
import platform
import torch
import argparse
from datetime import datetime
import numpy as np
import cv2
# =================Torch=======================
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import transform
# =================Local=======================
from lib.model.craft import CRAFT,CRAFT_MOB,CRAFT_LSTM
from lib.loss.mseloss import MSE_OHEM_Loss
from lib.dataloader.total import Total
from lib.dataloader.icdar import ICDAR
from lib.dataloader.icdar_video import ICDARV
from lib.dataloader.base import BaseDataset
import lib.dataloader.synthtext as syn80k
from lib.utils.img_hlp import cv_heatmap,np_img_normalize
from lib.fr_craft import CRAFTTrainer
from lib.config.train_default import cfg as tcfg
from lib.utils.log_hlp import save_image


__DEF_LOCAL_DIR = os.path.dirname(os.path.realpath(__file__))
__DEF_DATA_DIR = os.path.join(__DEF_LOCAL_DIR, 'dataset')
__DEF_CTW_DIR = os.path.join(__DEF_DATA_DIR, 'ctw')
__DEF_SVT_DIR = os.path.join(__DEF_DATA_DIR, 'svt', 'img')
__DEF_TTT_DIR = os.path.join(__DEF_DATA_DIR, 'totaltext')
__DEF_IC15_DIR = os.path.join(__DEF_DATA_DIR, 'ICDAR2015')
__DEF_ICV15_DIR = os.path.join(__DEF_DATA_DIR, 'ICDAR2015_video')
__DEF_SAVE_DIR = "/BACKUP/yom_backup/"

if(platform.system().lower()[:7]=='windows'):__DEF_SYN_DIR = "D:\\development\\SynthText"
elif(os.path.exists("/BACKUP/yom_backup/SynthText")):__DEF_SYN_DIR = "/BACKUP/yom_backup/SynthText"
else:__DEF_SYN_DIR = os.path.join(__DEF_DATA_DIR, 'SynthText')

if __name__ == "__main__":
    load_dir = "/home/yomcoding/Pytorch/MyResearch/saved_model/20201001-060925+craft_lstm.pkl"
    teacher_dir = "/home/yomcoding/Pytorch/MyResearch/pre_train/craft_mlt_25k.pkl"
    device='cuda'
    log_stp = 15
    max_s = 60
    max_vdo = 1
    net = torch.load(load_dir).float().to(device)
    teacher = torch.load(teacher_dir).float().to(device)
    net.eval()
    teacher.eval()
    net.init_state()
    train_dataset = ICDARV(os.path.join(__DEF_ICV15_DIR,'test'),include_name=True)
    current_step = 0
    for sample in train_dataset:
        samp_name = sample['name']
        vdo = sample['video']
        fps = sample['fps']
        frm_cnt = -1
        with torch.no_grad():
            while(True):
                ret, x = vdo.read()
                frm_cnt+=1

                if(ret==False):
                    break
                x = transform.resize(x,(640,640),preserve_range=True)
                if(len(x.shape)==3): 
                    x = np.expand_dims(x,0)

                nor_img = torch.from_numpy(np_img_normalize(x)).float().permute(0,3,1,2).to(device)
                pred,_ = net(nor_img)
                if(frm_cnt%log_stp):
                    # ensure the state is updated
                    continue
                if(max_s>0 and frm_cnt//fps>max_s):
                    break
                pred_t,_ = teacher(nor_img)
                pred = pred.to('cpu').numpy()
                pred_t = pred_t.to('cpu').numpy()

                ch = cv_heatmap(np.expand_dims(pred[0,0,:,:],-1))
                af = cv_heatmap(np.expand_dims(pred[0,1,:,:],-1))
                ch_t = cv_heatmap(np.expand_dims(pred_t[0,0,:,:],-1))
                af_t = cv_heatmap(np.expand_dims(pred_t[0,1,:,:],-1))

                frame = x[0].astype(np.uint8)
                frame = cv2.resize(frame,ch.shape[:2]).astype(np.uint8)
                lines = np.ones((frame.shape[0],5,3))*255.0
                lines = lines.astype(frame.dtype)

                img = np.concatenate((frame,lines,ch,lines,af),axis=-2)
                save_image(os.path.join(__DEF_SAVE_DIR,samp_name.split('.')[0],'{:05d}_img_ch_af_o.jpg'.format(frm_cnt)),img)
                img = np.concatenate((frame,lines,ch_t,lines,af_t),axis=-2)
                save_image(os.path.join(__DEF_SAVE_DIR,samp_name.split('.')[0],'{:05d}_img_ch_af_t.jpg'.format(frm_cnt)),img)
                net.lstmh = net.lstmh.detach()
                net.lstmc = net.lstmc.detach()
        vdo.release()
        current_step+=1
        if(max_vdo>0 and current_step>=max_vdo):
            break

            