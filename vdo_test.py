import os
import sys
import platform
import torch
import argparse
from datetime import datetime
import numpy as np
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
from lib.utils.img_hlp import cv_heatmap
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
__DEF_SAVE_DIR = os.path.join(__DEF_LOCAL_DIR, 'test')

if(platform.system().lower()[:7]=='windows'):__DEF_SYN_DIR = "D:\\development\\SynthText"
elif(os.path.exists("/BACKUP/yom_backup/SynthText")):__DEF_SYN_DIR = "/BACKUP/yom_backup/SynthText"
else:__DEF_SYN_DIR = os.path.join(__DEF_DATA_DIR, 'SynthText')

if __name__ == "__main__":
    load_dir = "/BACKUP/yom_backup/saved_model/CRAFT_LSTM_pre/20200921-181646+craft_LSTM_teacher.pkl"
    device='cuda'
    log_stp = 10

    net = torch.load(load_dir).float().to(device)
    net.eval()
    net.init_state()
    train_dataset = ICDARV(os.path.join(__DEF_ICV15_DIR,'test'),include_name=True)
    # for sample in train_dataset:
    sample = train_dataset[0]
    vdo = sample['video']
    

    frm_cnt = -1
    while(True):
        ret, x = vdo.read()
        frm_cnt+=1

        if(ret==False):
            break
        x = transform.resize(x,(640,640),preserve_range=True)
        if(len(x.shape)==3): 
            x = np.expand_dims(x,0)
        pred,_ = net(torch.from_numpy(x).float().permute(0,3,1,2).to(device))
        net.lstmh = net.lstmh.detach()
        net.lstmc = net.lstmc.detach()
        if(frm_cnt%log_stp):
            # ensure the state is updated
            continue
        pred = pred.detach().to('cpu')
        ch = np.expand_dims(pred[0,0].numpy(),-1)
        af = np.expand_dims(pred[0,1].numpy(),-1)
        
        save_image(os.path.join(__DEF_SAVE_DIR,'i_{}_{:05d}.jpg'.format(sample['name'].split('.')[0],frm_cnt)),x[0]/255.0)
        save_image(os.path.join(__DEF_SAVE_DIR,'c_{}_{:05d}.jpg'.format(sample['name'].split('.')[0],frm_cnt)),cv_heatmap(ch))
        save_image(os.path.join(__DEF_SAVE_DIR,'a_{}_{:05d}.jpg'.format(sample['name'].split('.')[0],frm_cnt)),cv_heatmap(af))