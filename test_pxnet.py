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
from lib.model.pixel_map import *
from lib.loss.mseloss import *
from lib.dataloader.total import Total
from lib.dataloader.icdar import *
from lib.dataloader.msra import MSRA
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
    DEF_BOOL_TRAIN_MASK = True
    DEF_BOOL_TRAIN_CE = True
    DEF_BOOL_TRAIN_BOX = False
    DEF_BOOL_LOG_LEVEL_MASK = False
    DEF_MASK_CH = 3
    DEF_CE_CH = 3
    DEF_BOX_CH = 5*2*2+1 # 10 points polygon (x,y) and 1 score map 
    top_ch=0
    if(DEF_BOOL_TRAIN_MASK):
        DEF_START_MASK_CH = top_ch
        top_ch+=DEF_MASK_CH
    if(DEF_BOOL_TRAIN_CE):
        DEF_START_CE_CH = top_ch
        top_ch+=DEF_CE_CH
    if(DEF_BOOL_TRAIN_BOX):
        DEF_START_BOX_CH = top_ch
        top_ch+=DEF_BOX_CH
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
        pred_np = pred.cpu().numpy()
        if(DEF_BOOL_TRAIN_MASK):
            bth_mask_np = pred_np[:,DEF_START_MASK_CH+0]
            bth_edge_np = pred_np[:,DEF_START_MASK_CH+1]
            bth_region_np = pred_np[:,DEF_START_MASK_CH+2]
        if(DEF_BOOL_TRAIN_CE):
            argmap = torch.argmax(pred[:,DEF_START_CE_CH:DEF_START_CE_CH+DEF_CE_CH],axis=1)
            bth_bin_ce_map = argmap.cpu().numpy().astype(np.uint8)
            bth_float_ce_map = (bth_bin_ce_map>0).astype(np.float32)
        
        for bthi in range(batch_size):
            image = bth_x[bthi].numpy().astype(np.uint8)
            
            region_np = bth_region_np[bthi]
            mask_np = bth_mask_np[bthi]
            edge_np = bth_edge_np[bthi]
            if(DEF_BOOL_TRAIN_CE):
                f_ce_map = bth_float_ce_map[bthi]
                region_np = np.where(region_np>0.4,region_np,0.0)
                region_np = np.where(region_np>f_ce_map,region_np,f_ce_map)

            boxes = bth_boxes[bthi]
            fname = img_names[bthi].split('.')[0]
            det_boxes, label_mask, label_list = cv_get_box_from_mask(region_np)
            boximage = image
            if(det_boxes.shape[0]>0):
                det_boxes = np_box_resize(det_boxes,mask_np.shape,image.shape[:-1],'polyxy')
                ids,precision,recall = cv_box_match(det_boxes,boxes,ovth=ovlap_th)
                boximage = cv_draw_poly(boximage,det_boxes,color=(0,255,0))
            else:
                precision=0 if(boxes.shape[0])else 1
                recall=0 if(boxes.shape[0])else 1

            precision_lst.append(precision)
            recall_lst.append(recall)
            if(recall<0.3 or precision<0.3):
                boximage = cv_draw_poly(boximage,boxes,color=(0,0,255))
                maskimage = cv_mask_image(image,cv_heatmap(region_np))
                cimg = concatenate_images([maskimage,mask_np,edge_np])
                save_image(os.path.join(save_dir,'{}_region_mask_edge.jpg'.format(fname)),cimg)
                save_image(os.path.join(save_dir,'{}_box.jpg'.format(fname)),boximage)
                if(DEF_BOOL_TRAIN_CE):
                    save_image(os.path.join(save_dir,'{}_label.jpg'.format(fname)),cv_labelmap(bth_bin_ce_map[bthi],2))

    precision_lst = np.array(precision_lst)
    recall_lst = np.array(recall_lst)
    recall_s = np.mean(recall_lst)
    precision_s = np.mean(precision_lst)
    f_s = recall_s*precision_s/(precision_s+recall_s)
    print("Recall {}".format(recall_s))
    print("Precision {}".format(precision_s))
    print("F-score {}".format(f_s))
        
if __name__ == "__main__":
    fdir = "/BACKUP/yom_backup/saved_model/PIX_Unet_MASK.pth"
    dataset = 'msra'
    work_dir = "/BACKUP/yom_backup/eval/pixtxt"
    work_dir = os.path.join(work_dir,dataset)
    batch_size = 8
    image_size = (640, 640)
    if(dataset=="ttt"):
        train_dataset = Total(
            os.path.join(DEF_TTT_DIR,'images','test'),
            os.path.join(DEF_TTT_DIR,'gt_pixel','test'),
            os.path.join(DEF_TTT_DIR,'gt_txt','test'),
            image_size=image_size,)
    elif(dataset=="msra"):
        train_dataset = MSRA(
            os.path.join(DEF_MSRA_DIR,'test'),
            os.path.join(DEF_MSRA_DIR,'test'),
            image_size=image_size)
    elif(dataset=="ic19"):
        train_dataset = ICDAR19(
            os.path.join(DEF_IC19_DIR,'images','test'),
            os.path.join(DEF_IC19_DIR,'gt_txt','test'),
            image_size=image_size)
    elif(dataset=="ic15"):
        train_dataset = ICDAR15(
            os.path.join(DEF_IC15_DIR,'images','test'),
            os.path.join(DEF_IC15_DIR,'gt_txt','test'),
            image_size=image_size)
    else:
        train_dataset = ICDAR13(
            os.path.join(DEF_IC13_DIR,'images','test'),
            os.path.join(DEF_IC13_DIR,'gt_txt','test'),
            os.path.join(DEF_IC13_DIR,'gt_pixel','test'),
            image_size=image_size,)
    x_input_function = train_dataset.x_input_function
    y_input_function = None

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=train_dataset.default_collate_fn,)
    model = PIX_Unet_MASK(min_map_ch=32,min_upc_ch=128,pretrained=False).float()
    model.load_state_dict(copyStateDict(torch.load(fdir)))
    model = model.cuda()
    test(model,train_loader,work_dir)
    pass