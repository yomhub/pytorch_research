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
from lib.model.pixel_map import PIX_TXT
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
from dirs import *

@torch.no_grad()
def test(net,dataloader,save_dir):
    ovlap_th = 0.5
    for o in net.state_dict().values():
        dev = o.device
        break
    
    precision_lst,recall_lst=[],[]
    for step,sample in enumerate(dataloader):
        # if(step>2):
        #     break
        bth_x = sample['image']
        bth_boxes = sample['box']
        img_names = sample['name']
        batch_size = bth_x.shape[0]
        x_nor = torch_img_normalize(bth_x.to(dev).float()).permute(0,3,1,2)
        pred,feat = net(x_nor)
        
        bth_mask_np = pred[:,0].cpu().numpy()
        bth_edge_np = pred[:,1].cpu().numpy()
        bth_region_np = pred[:,2].cpu().numpy()
        
        for bthi in range(batch_size):
            image = bth_x[bthi].numpy().astype(np.uint8)
            region_np = bth_region_np[bthi]
            mask_np = bth_mask_np[bthi]
            edge_np = bth_edge_np[bthi]
            boxes = bth_boxes[bthi]
            fname = img_names[bthi].split('.')[0]
            det_boxes, label_mask, label_list = cv_get_box_from_mask(region_np)
            boximage = image
            if(det_boxes.shape[0]>0):
                det_boxes = np_box_resize(det_boxes,mask_np.shape,image.shape[:-1],'polyxy')
                ids,precision,recall = cv_box_match(det_boxes,boxes,ovth=ovlap_th)
                boximage = cv_draw_poly(boximage,det_boxes,color=(0,255,0))
            else:
                precision=0
                recall=0

            precision_lst.append(precision)
            recall_lst.append(recall)
            if(recall<0.3 or precision<0.3):
                boximage = cv_draw_poly(boximage,boxes,color=(0,0,255))
                maskimage = cv_mask_image(image,cv_heatmap(region_np))
                cimg = concatenate_images([maskimage,mask_np,edge_np])
                save_image(os.path.join(save_dir,'region_mask_edge_{}.jpg'.format(fname)),cimg)
                save_image(os.path.join(save_dir,'boxes_{}.jpg'.format(fname)),boximage)
    precision_lst = np.array(precision_lst)
    recall_lst = np.array(recall_lst)
    print("Recall {}".format(np.mean(recall_lst)))
    print("Precision {}".format(np.mean(precision_lst)))
        
if __name__ == "__main__":
    fdir = "/BACKUP/yom_backup/saved_model/pixtxt.pkl.pth"
    work_dir = "/BACKUP/yom_backup/eval/pixtxt/total"
    batch_size = 8
    train_dataset = Total(
        os.path.join(DEF_TTT_DIR,'images','test'),
        os.path.join(DEF_TTT_DIR,'gt_pixel','test'),
        os.path.join(DEF_TTT_DIR,'gt_txt','test'),
        image_size=(1080, 1080),)

    x_input_function = train_dataset.x_input_function
    y_input_function = None

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=train_dataset.default_collate_fn,)
    model = PIX_TXT(pretrained=True).float()
    model.load_state_dict(copyStateDict(torch.load(fdir)))
    model = model.cuda()
    test(model,train_loader,work_dir)
    pass
            

            