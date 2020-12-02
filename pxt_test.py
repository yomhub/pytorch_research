import os
import sys
import platform
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
# =================Torch=======================
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# =================Local=======================
from lib.loss.mseloss import *
from lib.dataloader.total import Total
from lib.dataloader.icdar import *
from lib.dataloader.icdar_video import ICDARV
from lib.dataloader.minetto import Minetto
from lib.dataloader.base import BaseDataset
from lib.dataloader.synthtext import SynthText
from lib.utils.img_hlp import *
from lib.utils.log_hlp import *
from lib.config.train_default import cfg as tcfg
from lib.model.pixel_map import PIX_TXT
from dirs import *

@torch.no_grad()
def test(net,test_dataset,step):

    for datai,sample in enumerate(test_dataset):
        if(datai>step):
            break
        x = sample['image']
        xnor = torch.from_numpy(np.expand_dims(x,0)).float()
        xnor = torch_img_normalize(xnor).permute(0,3,1,2).cuda()
        pred,feat = net(xnor)

        pred_mask = pred[0,0].to('cpu').numpy()
        pred_edge = pred[0,1].to('cpu').numpy()
        pred_mask = (pred_mask*255.0).astype(np.uint8)
        pred_edge = (pred_edge*255.0).astype(np.uint8)
        pred_mask = np.stack([pred_mask,pred_mask,pred_mask],-1)
        pred_edge = np.stack([pred_edge,pred_edge,pred_edge],-1)
        smx = cv2.resize(x.astype(np.uint8),(pred_edge.shape[1],pred_edge.shape[0]))
        line = np.ones((smx.shape[0],3,3),dtype=smx.dtype)*255
        img = np.concatenate((smx,line,pred_mask,line,pred_edge),-2)
        
        save_image(os.path.join(DEF_WORK_DIR,'eval',"{}.jpg".format(sample['name'])),img)

if __name__ == "__main__":
    loddir="/home/yomcoding/Pytorch/MyResearch/saved_model/pixtxt.pth"
    model = PIX_TXT(pretrained=True,load_mobnet=loddir)
    model.load_state_dict(copyStateDict(torch.load(loddir)))
    model = model.float().cuda()
    test_dataset = Total(
        os.path.join(DEF_TTT_DIR,'images','test'),
        os.path.join(DEF_TTT_DIR,'gt_pixel','test'),
        os.path.join(DEF_TTT_DIR,'gt_txt','test'),
        image_size=(1080, 1080),)
    test(model,test_dataset,10)
    pass
