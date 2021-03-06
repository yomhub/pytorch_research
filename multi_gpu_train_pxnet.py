import os
import sys
import platform
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
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
from lib.model.siamfc import SiamesePXT,SiameseMask,conv2d_dw_group,SiameseCRAFT
from lib.loss.mseloss import *
from lib.dataloader.total import Total
from lib.dataloader.icdar import *
from lib.dataloader.msra import MSRA
from lib.dataloader.minetto import Minetto
from lib.dataloader.base import split_dataset_cls_to_train_eval
from lib.dataloader.synthtext import SynthText
from lib.utils.img_hlp import *
from lib.utils.log_hlp import *
from lib.utils.net_hlp import adjust_learning_rate
from lib.config.train_default import cfg as tcfg
from dirs import *

DEF_MAX_TRACKING_BOX_NUMBER = 1
# pixel mask, edge mask, region mask, threshold mask
DEF_MASK_CH = 4
# [bg, fg, boundary] | [text bg, text fg]
DEF_BOOL_MULTILEVEL_CE = False
DEF_CE_CH = 3+2

DEF_BOOL_POLY_REGRESSION = False
DEF_BOX_HITMAP_CH = 2
DEF_POLY_NUM = 10
#polygon (x,y) or (dcx,dcy,w,h)
DEF_BOX_CH = DEF_POLY_NUM*2 if(DEF_BOOL_POLY_REGRESSION)else 4

DEF_BOOL_TRAIN_TXT_MASK = True

def train(rank, world_size, args):
    """
    Custom training function, return 0 for join operation
    args:
        rank: rank id [0-N-1] for N gpu
        world_size: N, total gup number
        args: argparse from input
    """
    if(rank == 0):
        log = sys.stdout
    image_size=(640, 640)
    random_b = bool(args.random)
    time_start = datetime.now()
    work_dir = DEF_WORK_DIR
    work_dir = os.path.join(work_dir,'log')
    if(args.name):
        work_dir = os.path.join(work_dir,args.name)
    if(rank==0 and not(args.debug)):
        logger = SummaryWriter(os.path.join(work_dir,time_start.strftime("%Y%m%d-%H%M%S")))
    else:
        logger = None

    batch_size = args.batch
    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    dev = 'cuda:{}'.format(rank)

    DEF_BOOL_TRAIN_MASK = False
    DEF_BOOL_TRAIN_CE = False
    DEF_BOOL_TRAIN_BOX = False
    DEF_BOOL_LOG_LEVEL_MASK = False
    net_args = "Save dir: {}, net: {}.\n".format(args.save,args.net)

    args.net = args.net.lower()
    if(args.net=='vgg_pur_cls'):
        model = VGG_PUR_CLS(include_b0=True,padding=False,pretrained=True).float()
        net_args+="include_b0={},padding={},pretrained={}\n".format(True,False,True)
        DEF_BOOL_TRAIN_CE = True
    elif(args.net=='vggunet_pxmask'):
        model = VGGUnet_PXMASK(include_b0=True,padding=False,pretrained=True).float()
        net_args+="include_b0={},padding={},pretrained={}\n".format(True,False,True)
        DEF_BOOL_TRAIN_MASK = True
    elif(args.net=='vgg_pxmask'):
        model = VGG_PXMASK(include_b0=True,padding=False,pretrained=True).float()
        net_args+="include_b0={},padding={},pretrained={}\n".format(True,False,True)
        DEF_BOOL_TRAIN_MASK = True
        DEF_BOOL_LOG_LEVEL_MASK = True
    elif(args.net=='pix_txt'):
        model = PIX_TXT().float()
        DEF_BOOL_TRAIN_MASK = True
        DEF_BOOL_LOG_LEVEL_MASK = True
    elif('pix_unet' in args.net):
        net_args_dict = {}
        if('mask' in args.net):
            net_args_dict['mask_ch']=DEF_MASK_CH
            net_args_dict['min_mask_ch']=32
            net_args+="mask_ch={},min_mask_ch={},".format(DEF_MASK_CH,32)
            DEF_BOOL_TRAIN_MASK = True
        if('txt' not in args.net):
            DEF_BOOL_TRAIN_TXT_MASK = False

        if('cls' in args.net):
            net_args_dict['cls_ch']=DEF_CE_CH
            net_args_dict['min_cls_ch']=32
            net_args_dict['multi_level']=DEF_BOOL_MULTILEVEL_CE
            net_args+="cls_ch={},multi_level={},min_cls_ch={}\n".format(DEF_CE_CH,DEF_BOOL_MULTILEVEL_CE,32)
            DEF_BOOL_TRAIN_CE = True

        if('box' in args.net):
            net_args_dict['box_ch']=DEF_BOX_CH
            net_args_dict['hitmap_ch']=DEF_BOX_HITMAP_CH
            net_args_dict['min_box_ch']=32
            net_args+="box_ch={},hitmap_ch={},min_box_ch={}\n".format(DEF_BOX_CH,DEF_BOX_HITMAP_CH,32)
            DEF_BOOL_TRAIN_BOX = True

        net_args+="include_final={},basenet=\'{}\',min_upc_ch={},pretrained={}\n".format(args.have_fc,args.basenet,128,False)
        model = PIX_Unet(include_final=args.have_fc,basenet=args.basenet,min_upc_ch=128,pretrained=True,**net_args_dict).float()
    else:
        model = PIX_MASK().float()
        DEF_BOOL_TRAIN_MASK = True
        DEF_BOOL_TRAIN_CE = True
        DEF_BOOL_LOG_LEVEL_MASK = True
    
    net_args+="\n\tDEF_BOOL_TRAIN_MASK={},DEF_BOOL_TRAIN_TXT_MASK={},\n\tDEF_BOOL_TRAIN_CE={},\n\tDEF_BOOL_TRAIN_BOX={}".format(
        DEF_BOOL_TRAIN_MASK,DEF_BOOL_TRAIN_TXT_MASK,
        DEF_BOOL_TRAIN_CE,DEF_BOOL_TRAIN_BOX
    )
    if(not args.debug and rank==0 and args.save):
        fname = args.save.split('.')[0]+'_args.txt'
        os.makedirs(os.path.dirname(fname),exist_ok=True)
        arglog = open(fname,'w')
        arglog.write(net_args)
        arglog.close()

    DEF_BOOL_TRACKING = False
    if(args.tracker=='siamesemask'):
        model = SiameseMask(basenet=model).float()
        DEF_BOOL_TRACKING = True
    elif(args.tracker=='siamesecraft'):
        model = SiameseCRAFT(basenet=model,feature_chs=model.final_ch).float()
        DEF_BOOL_TRACKING = True

    if(args.load and os.path.exists(args.load)):
        model.load_state_dict(copyStateDict(torch.load(args.load)))
    model = model.cuda(rank)

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

    # define loss function (criterion) and optimizer
    criterion = MSE_2d_Loss(pixel_sum=False).cuda(rank)
    criterion_ce = nn.CrossEntropyLoss(ignore_index = -1)
    if(os.path.exists(args.opt)):
        optimizer = torch.load(args.opt)
    elif(args.opt.lower()=='adam'):
        optimizer = optim.Adam(model.parameters(), lr=args.learnrate, weight_decay=tcfg['OPT_DEC'])
    elif(args.opt.lower() in ['adag','adagrad']):
        optimizer = optim.Adagrad(model.parameters(), lr=args.learnrate, weight_decay=tcfg['OPT_DEC'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learnrate, momentum=tcfg['MMT'], weight_decay=tcfg['OPT_DEC'])
    if(args.lr_decay):
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #     mode='min',factor=0.9,
        #     patience=100,cooldown=30,
        #     min_lr=0.0001,
        #     verbose=True
        #     )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    if(world_size>1):
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    if(args.dataset=="ttt"):
        train_dataset = Total(
            os.path.join(DEF_TTT_DIR,'images','train'),
            os.path.join(DEF_TTT_DIR,'gt_pixel','train'),
            os.path.join(DEF_TTT_DIR,'gt_txt','train'),
            image_size=image_size,
            # include_bg=True,
            )
        eval_dataset = Total(
            os.path.join(DEF_TTT_DIR,'images','test'),
            os.path.join(DEF_TTT_DIR,'gt_pixel','test'),
            os.path.join(DEF_TTT_DIR,'gt_txt','test'),
            image_size=image_size,
            include_bg=True,
            )
    elif(args.dataset=="msra"):
    # ONLY PROVIDE SENTENCE LEVEL BOX
        train_dataset = MSRA(
            os.path.join(DEF_MSRA_DIR,'train'),
            os.path.join(DEF_MSRA_DIR,'train'),
            image_size=image_size)
        eval_dataset = MSRA(
            os.path.join(DEF_MSRA_DIR,'test'),
            os.path.join(DEF_MSRA_DIR,'test'),
            image_size=image_size)
    elif(args.dataset=="ic19"):
        # train_dataset = ICDAR19(
        #     os.path.join(DEF_IC19_DIR,'images','train'),
        #     os.path.join(DEF_IC19_DIR,'gt_txt','train'),
        #     image_size=image_size)
        # eval_dataset = ICDAR19(
        #     os.path.join(DEF_IC19_DIR,'images','test'),
        #     os.path.join(DEF_IC19_DIR,'gt_txt','test'),
        #     image_size=image_size,)
        train_dataset,eval_dataset = split_dataset_cls_to_train_eval(ICDAR19,0.2,
            os.path.join(DEF_IC19_DIR,'images','train'),
            os.path.join(DEF_IC19_DIR,'gt_txt','train'),
            image_size=image_size)
    elif(args.dataset=="ic15"):
        train_dataset = ICDAR15(
            os.path.join(DEF_IC15_DIR,'images','train'),
            os.path.join(DEF_IC15_DIR,'gt_txt','train'),
            image_size=image_size)
        eval_dataset = ICDAR15(
            os.path.join(DEF_IC15_DIR,'images','test'),
            os.path.join(DEF_IC15_DIR,'gt_txt','test'),
            image_size=image_size)
    else:
        train_dataset = ICDAR13(
            os.path.join(DEF_IC13_DIR,'images','train'),
            os.path.join(DEF_IC13_DIR,'gt_txt','train'),
            # os.path.join(DEF_IC13_DIR,'gt_pixel','train'),
            image_size=image_size,)
        eval_dataset = ICDAR13(
            os.path.join(DEF_IC13_DIR,'images','test'),
            os.path.join(DEF_IC13_DIR,'gt_txt','test'),
            # os.path.join(DEF_IC13_DIR,'gt_pixel','test'),
            image_size=image_size)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               collate_fn=train_dataset.default_collate_fn,)
    if(args.eval):
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,
                                                                        num_replicas=world_size,
                                                                        rank=rank)
        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True,
                                                sampler=eval_sampler,
                                                collate_fn=eval_dataset.default_collate_fn,)
    time_c = datetime.now()
    total_step = len(train_loader)
    kernel = np.ones((3,3),dtype=np.uint8)
    all_recall, all_precision = [],[]
    last_max_recall, last_max_precision, last_max_fscore = 0.0,0.0,0.0

    for epoch in range(args.epoch):
        epoch_loss_list = []
        for stepi, sample in enumerate(train_loader):
            x = sample['image']
            b_have_mask = bool('mask' in sample)
            if(b_have_mask):
                mask = sample['mask']

            boxes = sample['box']
            if(random_b):
                boxest_list = []
                xt_list = []
                image = x.numpy()
                if(b_have_mask):
                    maskt_list = []
                    mask_np = mask.numpy()
                
                for i in range(image.shape[0]):
                    if(not isinstance(boxes,type(None)) and boxes[i].shape[0]>0):
                        imgt,boxt,M = cv_random_image_process(image[i],boxes[i],np.random.random()>0.5)
                    else:
                        imgt,boxt,M = cv_random_image_process(image[i],None,np.random.random()>0.5)

                    xt_list.append(imgt)
                    boxest_list.append(boxt if(not isinstance(boxt,type(None)))else boxes[i])
                    if(b_have_mask):
                        maskt_list.append(cv2.warpAffine(mask_np[i], M[:-1], (image.shape[2],image.shape[1])))

                boxes = boxest_list
                image = np.stack(xt_list)
                x = torch.from_numpy(image)
                if(b_have_mask):
                    mask_np = np.stack(maskt_list).astype(np.uint8)
                    if(len(mask_np.shape)==3):
                        mask_np = np.expand_dims(mask_np,-1)
                    mask = torch.from_numpy(mask_np)

            xnor = x.float().cuda(non_blocking=True)
            xnor = torch_img_normalize(xnor).permute(0,3,1,2)
            
            # Forward pass
            pred,feat = model(xnor)
            loss_dict = {}
            if(DEF_BOOL_TRAIN_MASK):
                if(b_have_mask):
                    if(mask.shape[-1]==3): mask = mask[:,:,:,0:1]
                    mask_np = mask.numpy()[:,:,:,0]
                    mask_np = np.where(mask_np>0,255,0).astype(np.uint8)
                    edge_np = [cv2.dilate(cv2.Canny(bth,100,200),kernel) for bth in mask_np]
                    # np (batch,h,w,1)-> torch (batch,1,h,w)
                    edge_np = np.stack(edge_np,0)
                                       
                elif(args.genmask):
                    # weak surpress learning
                    mask_list = []
                    edge_list = []
                    gen_mask_prob_list = []
                    if(DEF_BOOL_TRAIN_CE and DEF_CE_CH>3):
                        # have text ce
                        labels= torch.argmax(pred[:,DEF_START_CE_CH+3:DEF_START_CE_CH+DEF_CE_CH],dim=1)
                        np_pred_mask = (labels==1).cpu().detach().numpy()
                        np_pred_mask = np.logical_or(np_pred_mask,(pred[:,DEF_START_MASK_CH+0]>0.3).cpu().detach().numpy())
                    else:
                        # region mask
                        np_pred_mask = (pred[:,DEF_START_MASK_CH+0]>0.3).cpu().detach().numpy()

                    x_np = x.numpy().astype(np.uint8)
                    for batchi in range(x_np.shape[0]):
                        bth_mask,gen_mask_prob  = cv_gen_binary_map_by_pred(x_np[batchi],boxes[batchi],np_pred_mask[batchi])
                        bth_edge = cv2.dilate(cv2.Canny(bth_mask,100,200),kernel)
                        mask_list.append(bth_mask)
                        edge_list.append(bth_edge)
                        gen_mask_prob_list.append(gen_mask_prob)

                    mask_np = np.stack(mask_list,0)
                    edge_np = np.stack(edge_list,0)
                    gen_mask_prob_np = np.stack(gen_mask_prob_list,0)
                    gen_mask_prob = torch.from_numpy(np.expand_dims(gen_mask_prob_np,-1)).cuda(non_blocking=True)
                    gen_mask_prob = gen_mask_prob.float().permute(0,3,1,2)
                
                if(DEF_BOOL_TRAIN_TXT_MASK and (b_have_mask or args.genmask)):
                    edge = torch.from_numpy(np.expand_dims(edge_np,-1)).cuda(non_blocking=True)
                    edge = (edge>0).float().permute(0,3,1,2)
                    if(len(mask_np.shape)==3):
                        mask = torch.from_numpy(np.expand_dims(mask_np,-1)).cuda(non_blocking=True)
                    else:
                        mask = torch.from_numpy(mask_np).cuda(non_blocking=True)
                    mask = (mask>0).float().permute(0,3,1,2)
                    if(b_have_mask):
                        loss_dict['txt_loss'] = criterion(pred[:,DEF_START_MASK_CH+0], mask)
                        loss_dict['edge_loss'] = criterion(pred[:,DEF_START_MASK_CH+1], edge)
                    elif(args.genmask):
                        loss_dict['txt_loss'] = criterion(pred[:,DEF_START_MASK_CH+0], mask, gen_mask_prob)
                        loss_dict['edge_loss'] = criterion(pred[:,DEF_START_MASK_CH+1], edge, gen_mask_prob)

            region_mask_np = []
            region_mask_bin_np = []
            boundary_mask_bin_np = []
            for batchi,batch_boxes in enumerate(boxes):
                # (h,w)
                if(batch_boxes.shape[0]==0):
                    gas_map = np.zeros(x.shape[-3:-1],dtype=np.float32)
                    blmap = np.zeros(pred.shape[-2:],dtype=np.uint8)
                    boundadrymap = np.zeros(pred.shape[-2:],dtype=np.uint8)
                else:
                    if(b_have_mask and args.bxrefine):
                        batch_mask = mask_np[batchi]
                        if(len(batch_mask.shape)==3):
                            batch_mask=batch_mask[:,:,0]
                        batch_boxes = cv_refine_box_by_binary_map(batch_boxes,batch_mask)
                    gas_map,blmap = cv_gen_gaussian_by_poly(batch_boxes,x.shape[-3:-1],return_mask=True)
                    blmap = np.where(blmap>0,255,0).astype(np.uint8)
                    blmap = cv2.resize(blmap,(pred.shape[-1],pred.shape[-2]))
                    # slightly shrink the region to enhance boundary
                    blmap = cv2.erode(blmap,kernel,iterations=1)
                    boundadrymap = blmap-cv2.erode(blmap,kernel,iterations=2)
        
                region_mask_np.append(gas_map)
                region_mask_bin_np.append(blmap)
                boundary_mask_bin_np.append(boundadrymap)

            if(DEF_BOOL_TRAIN_MASK):
                # based on box and optional text mask
                region_mask_np = np.expand_dims(np.stack(region_mask_np,0),-1).astype(np.float32)
                region_mask = torch.from_numpy(region_mask_np).cuda(non_blocking=True).permute(0,3,1,2)
                loss_dict['region_loss'] = criterion(pred[:,DEF_START_MASK_CH+2], region_mask)
                if(DEF_BOOL_TRAIN_TXT_MASK and (b_have_mask or args.genmask) and DEF_MASK_CH>3):
                    img = x.numpy().astype(np.uint8)
                    threshold_np = np.stack([cv2.cvtColor(o,cv2.COLOR_RGB2GRAY) for o in img],0)
                    threshold_np[mask_np==0]=0
                    if(len(threshold_np.shape)==3):
                        threshold_np = np.expand_dims(threshold_np,-1)
                    threshold = torch.from_numpy(threshold_np).float().cuda(non_blocking=True).permute(0,3,1,2)
                    threshold /= 255.0
                    loss_dict['threshold_loss'] = criterion(pred[:,DEF_START_MASK_CH+3],threshold)

            if(DEF_BOOL_TRAIN_CE):
                # based on region mask
                region_mask_bin_np = np.stack(region_mask_bin_np,0).astype(np.uint8)
                boundary_mask_bin_np = np.stack(boundary_mask_bin_np,0).astype(np.uint8)

                ce_y = np.zeros(region_mask_bin_np.shape,dtype=np.int64)
                ce_y[region_mask_bin_np>0]=1
                ce_y[boundary_mask_bin_np>0]=2
                ce_y_torch = torch.from_numpy(ce_y).cuda(non_blocking=True)
                loss_dict['region_ce_loss'] = criterion_ce(pred[:,DEF_START_CE_CH:DEF_START_CE_CH+3], ce_y_torch)
                # text binarization mask, based on text mask
                if(DEF_BOOL_TRAIN_TXT_MASK and (b_have_mask or args.genmask) and DEF_CE_CH>3):
                    mask_ce = F.interpolate(mask,size=pred.shape[2:], mode='bilinear', align_corners=False)
                    mask_ce = (mask_ce[:,0]>0).type(torch.int64)
                    loss_dict['txt_ce_loss'] = criterion_ce(pred[:,DEF_START_CE_CH+3:DEF_START_CE_CH+DEF_CE_CH], mask_ce)

            if(DEF_BOOL_TRAIN_BOX):
                pred_bx = pred[:,DEF_START_BOX_CH+DEF_BOX_HITMAP_CH:DEF_START_BOX_CH+DEF_BOX_CH+DEF_BOX_HITMAP_CH]
                div_num = DEF_POLY_NUM//2
                bx_loss_lst=[]
                small_image_size_xy = np.array([pred_bx.shape[-1],pred_bx.shape[-2]])
                box_oneheat_map =[]
                for batchi in range(x.shape[0]):
                    batch_boxes = boxes[batchi]
                    ct_box_oneheat_map = np.zeros(pred_bx.shape[-2:],dtype=np.float32)
                    if(batch_boxes.shape[0]==0):
                        box_oneheat_map.append(ct_box_oneheat_map)
                        continue
                    boxes_small = np_box_resize(batch_boxes,x.shape[1:3],pred_bx.shape[-2:],'polyxy')
                    ct_box = np_polybox_center(boxes_small)
                    rect_box = np_polybox_minrect(boxes_small)
                    ct_rect_box = np_polybox_center(rect_box)
                    for boxi in range(boxes_small.shape[0]):
                        if(boxes_small[boxi].shape[0]<4):
                            continue
                        cx,cy = ct_box[boxi].astype(np.uint16)
                        if(cx>=pred_bx.shape[-1] or cy>=pred_bx.shape[-2] or cx<0 or cy<0):
                            continue
                        ct_box_oneheat_map[cy:cy+2,cx:cx+2]=1.0
                        if(DEF_BOOL_POLY_REGRESSION):
                            poly_gt = np_split_polygon(boxes_small[boxi],div_num)
                            poly_gt = poly_gt.reshape(-1,2)
                            if(poly_gt.size!=DEF_POLY_NUM*2):
                                continue
                            bx_gt_nor = (poly_gt-ct_box[boxi])/small_image_size_xy
                            bx_gt_nor = bx_gt_nor.reshape(-1)
                        else:
                            w,h = (rect_box[boxi,2]-rect_box[boxi,0])/small_image_size_xy
                            det_cx,det_cy = (ct_rect_box[boxi]-ct_box[boxi])/small_image_size_xy
                            bx_gt_nor = np.array([det_cx,det_cy,w,h])

                        bx_gt_nor = torch.from_numpy(bx_gt_nor).float().cuda(non_blocking=True)
                        bx_loss_lst.append(torch.mean(torch.abs(pred_bx[batchi,:,cy-1,cx-1]-bx_gt_nor)))
                    box_oneheat_map.append(ct_box_oneheat_map)

                if(bx_loss_lst):
                    loss_dict['bx_loss'] = sum(bx_loss_lst)/len(bx_loss_lst)
                    box_oneheat_map = np.expand_dims(np.stack(box_oneheat_map,0),-1)
                    box_oneheat_map = torch.from_numpy(box_oneheat_map).float().cuda(non_blocking=True).permute(0,3,1,2)
                    if(DEF_BOX_HITMAP_CH==1):
                        loss_dict['bx_hit_loss'] = criterion(pred[:,DEF_START_BOX_CH:DEF_START_BOX_CH+1],box_oneheat_map)
                    else:
                        bx_hit_ce = F.interpolate(box_oneheat_map,size=pred.shape[2:], mode='bilinear', align_corners=False)
                        loss_dict['bx_hit_ce_loss'] = criterion_ce(pred[:,DEF_START_BOX_CH:DEF_START_BOX_CH+DEF_BOX_HITMAP_CH],
                            (box_oneheat_map[:,0]>0).type(torch.int64))
                    
            if(DEF_BOOL_TRACKING):
                tracking_loss_lst = []
                for batchi in range(x.shape[0]):
                    batch_boxes = boxes[batchi]
                    if(batch_boxes.shape[0]==0):
                        continue
                    batch_x_np = x[batchi].numpy()
                    batch_recbox = np_polybox_minrect(batch_boxes,'polyxy')
                    ws = np.linalg.norm(batch_recbox[:,0]-batch_recbox[:,1],axis=-1)
                    hs = np.linalg.norm(batch_recbox[:,0]-batch_recbox[:,3],axis=-1)
                    inds = np.argsort(ws*hs)[::-1]
                    slc_boxes = batch_boxes[inds[:DEF_MAX_TRACKING_BOX_NUMBER]]
                    slc_recbox = batch_recbox[inds[:DEF_MAX_TRACKING_BOX_NUMBER]]
                    tracking_loss = 0.0
                    cnt=0
                    
                    for sbxi,sig_sub_box in enumerate(slc_boxes):
                        w_min,h_min=230,230
                        w_max,h_max=480,480
                        sub_img,_,polymask = cv_crop_image_by_polygon(batch_x_np,sig_sub_box,return_mask=True)
                        h_sub,w_sub = sub_img.shape[:-1]
                        if(sub_img.shape[0]<h_min or sub_img.shape[1]<w_min or sub_img.shape[1]>w_max or sub_img.shape[0]>h_max):
                            sub_img_nor = cv2.resize(sub_img,
                                (min(w_max,max(w_min,sub_img.shape[1])),min(h_max,max(h_min,sub_img.shape[0]))))
                        else:
                            sub_img_nor = sub_img

                        sub_img_nor = torch.from_numpy(np.expand_dims(sub_img_nor,0)).float()
                        sub_img_nor = torch_img_normalize(sub_img_nor).cuda(non_blocking=True).permute(0,3,1,2)
                        try:
                        # if(1):
                            pred_sub,feat_sub = model(sub_img_nor)
                            feat_sub = F.interpolate(feat_sub,size=(h_sub//2,w_sub//2), mode='bilinear', align_corners=False)
                            # feat_obj = (feat_sub.b1,feat_sub.b2,feat_sub.b3,feat_sub.b4)
                            # feat_search = (feat.b1[batchi:batchi+1],feat.b2[batchi:batchi+1],feat.b3[batchi:batchi+1],feat.b4[batchi:batchi+1])
                            feat_search = feat[batchi:batchi+1]
                            # nn.parallel.DistributedDataParallel.module to call orginal class
                            match_map,feat_m = model.match(feat_sub,feat_search)
                            sub_ch_mask = cv_gen_gaussian_by_poly(
                                np_box_resize(sig_sub_box,x.shape[1:3],match_map.shape[2:],'polyxy'),
                                match_map.shape[2:]
                            )
                            sub_ch_mask_torch = torch.from_numpy(sub_ch_mask.reshape((1,sub_ch_mask.shape[0],sub_ch_mask.shape[1],1))).float().cuda(non_blocking=True).permute(0,3,1,2)
                            tracking_loss+=criterion(match_map,sub_ch_mask_torch)

                            cnt+=1
                        except Exception as e:
                            sys.stdout.write("Err at {}, X.shape: {}, sub_img shape {}:\n".format(sample['name'][sbxi],batch_x_np.shape,sub_img_nor.shape))
                            sys.stdout.write("\t{}\n".format(str(e)))
                            sys.stdout.flush()
                            continue
                    if(cnt>0):
                        tracking_loss_lst.append(tracking_loss/cnt)
                if(tracking_loss_lst):
                    loss_dict['tracking']=torch.mean(torch.stack(tracking_loss_lst))
            loss = 0.0
            for keyn,value in loss_dict.items():
                loss+=value
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_list.append(loss.item())

            if(logger and rank==0):
                logger.add_scalar('Loss/total loss', loss.item(), epoch*total_step+stepi)
                for keyn,value in loss_dict.items():
                    logger.add_scalar('Loss/{}'.format(keyn), value.item(), epoch*total_step+stepi)
                if(args.lr_decay):
                    logger.add_scalar('LR rate', optimizer.param_groups[0]['lr'], epoch*total_step+stepi)
                logger.flush()
        
        # ==============
        # End of epoch
        # ==============
        if(args.lr_decay):
            epoch_loss = np.array(epoch_loss_list)
            mean_epoch_loss = np.mean(epoch_loss)
            var_epoch_loss = np.var(epoch_loss)
            scheduler.step(loss)
            scheduler.step()

        if(args.eval):
            ovlap_th = 0.5
            recall_list,precision_list = [],[]
            recall_list_2x,precision_list_2x = [],[]
            recall_list_4x,precision_list_4x = [],[]
            mask_loss_list = []
            mask_ce_loss_list = []
            mask_box_map_list = []
            mask_box_ce_map_list = []
            bool_hit_max,bool_low_fscore = False,False
            log_eval_num = 5
            rng = np.random.default_rng()
            log_eval_id = rng.choice(len(eval_loader)-1, size=log_eval_num, replace=False)
            eval_img_log_cnt = 0
            log_images_max_cnt = 10 if((epoch+1)%10==0)else -1
            with torch.no_grad():
                for stepi, sample in enumerate(eval_loader):
                    log_images = {}
                    eva_bth_x = sample['image']
                    eva_boxes = sample['box']
                    eva_xnor = eva_bth_x.float().cuda(non_blocking=True)
                    eva_xnor = torch_img_normalize(eva_xnor).permute(0,3,1,2)
                    eva_pred,eva_feat = model(eva_xnor)
                    eva_pred_np = eva_pred.cpu().numpy()
                    eva_small_boxes = [np_box_resize(o,eva_bth_x.shape[1:3],eva_pred.shape[2:4],'polyxy') if(not isinstance(o,type(None)) and o.shape[0]>0)else None for o in eva_boxes]
                    # resized feature proformance 
                    if(DEF_BOOL_TRAIN_MASK):
                        fupb0 = eva_feat.upb0
                        fupb0_2x = F.interpolate(fupb0, size=(fupb0.shape[-2]//2,fupb0.shape[-1]//2), mode='bilinear', align_corners=False)
                        fupb0_2x = F.interpolate(fupb0_2x, size=fupb0.shape[-2:], mode='bilinear', align_corners=False)
                        fupb0_4x = F.interpolate(fupb0, size=(fupb0.shape[-2]//4,fupb0.shape[-1]//4), mode='bilinear', align_corners=False)
                        fupb0_4x = F.interpolate(fupb0_4x, size=fupb0.shape[-2:], mode='bilinear', align_corners=False)
                        if(world_size>1):
                            eva_pred_2x = model.module.mask(fupb0_2x)
                            eva_pred_4x = model.module.mask(fupb0_4x)
                        else:
                            eva_pred_2x = model.mask(fupb0_2x)
                            eva_pred_4x = model.mask(fupb0_4x)
                        eva_pred_2x_np = eva_pred_2x.cpu().numpy()
                        eva_pred_4x_np = eva_pred_4x.cpu().numpy()
                    # prepare data
                    if(DEF_BOOL_TRAIN_CE):
                        # process hold batch to speed up
                        argmap = torch.argmax(eva_pred[:,DEF_START_CE_CH:DEF_START_CE_CH+3],axis=1)
                        eva_bth_bin_ce_map = (argmap==1).cpu().numpy().astype(np.uint8)

                    # log image
                    if(log_images_max_cnt>0 and rank==0):
                        if(DEF_BOOL_TRAIN_MASK):
                            log_images['image']=cv2.resize(eva_bth_x[0].numpy(),(eva_pred.shape[3],eva_pred.shape[2]))
                            log_images['region_mask']=cv_heatmap(eva_pred_np[0,DEF_START_MASK_CH+2])
                            log_images['region_mask_2x']=cv_heatmap(eva_pred_2x_np[0,DEF_START_MASK_CH+2])
                            log_images['region_mask_4x']=cv_heatmap(eva_pred_4x_np[0,DEF_START_MASK_CH+2])
                            if(DEF_BOOL_TRAIN_TXT_MASK):
                                tmp = (eva_pred_np[0,DEF_START_MASK_CH+0]*255).astype(np.uint8)
                                log_images['text_mask']=np.stack([tmp,tmp,tmp],-1)
                                tmp = (eva_pred_np[0,DEF_START_MASK_CH+1]*255).astype(np.uint8)
                                log_images['edge_mask']=np.stack([tmp,tmp,tmp],-1)
                                if(DEF_MASK_CH>3):
                                    tmp = (eva_pred_np[0,DEF_START_MASK_CH+3]*255).astype(np.uint8)
                                    log_images['threshold_mask']=np.stack([tmp,tmp,tmp],-1)

                        if(DEF_BOOL_TRAIN_CE):
                            log['argmap'].append(argmap[0].cpu().numpy().astype(np.uint8))
                            if(DEF_BOOL_TRAIN_TXT_MASK and (DEF_CE_CH-3)>=2):
                                log['text_argmap'].append(
                                    torch.argmax(eva_pred[0,DEF_START_CE_CH+3:DEF_START_CE_CH+5],axis=0).cpu().numpy().astype(np.uint8)
                                )

                    # calculate mask difference
                    if((DEF_BOOL_TRAIN_MASK or DEF_BOOL_TRAIN_CE) and 'mask' in sample):
                        # eval_mask = sample['mask'].cuda(non_blocking=True)
                        # eval_mask = (eval_mask>0).float()
                        # if(len(eval_mask.shape)==4):
                        #     eval_mask = eval_mask.permute(0,3,1,2)
                        # eval_mask_loss = criterion(eva_pred[:,DEF_START_MASK_CH+0], eval_mask)
                        # mask_loss_list.append(eval_mask_loss)

                        # global mAP 
                        eva_gt_mask_bin_np = sample['mask'].numpy().astype(np.uint8)
                        if(len(eva_gt_mask_bin_np.shape)==4):
                            eva_gt_mask_bin_np=eva_gt_mask_bin_np[:,:,:,0]
                        eva_gt_mask_bin_np = [cv2.resize(o,(eva_pred.shape[3],eva_pred.shape[2])) for o in eva_gt_mask_bin_np]
                        eva_gt_mask_bin_np = np.stack(eva_gt_mask_bin_np)
                        if(DEF_BOOL_TRAIN_MASK and DEF_BOOL_TRAIN_TXT_MASK):
                            eva_pred_mask_bin_np = (eva_pred_np[:,DEF_START_MASK_CH+0]*255).astype(np.uint8)
                            mask_loss_list.append(np.sum(eva_pred_mask_bin_np==eva_gt_mask_bin_np)/np.sum(eva_gt_mask_bin_np>0))

                        if(DEF_BOOL_TRAIN_CE and DEF_CE_CH>3):
                            labels = torch.argmax(eva_pred[:,DEF_START_CE_CH+3:DEF_START_CE_CH+DEF_CE_CH],dim=1)
                            labels = labels.cpu().detach().numpy()
                            labels = np.where(labels==1,255,0).astype(np.uint8)
                            mask_ce_loss_list.append(np.sum(labels==eva_gt_mask_bin_np)/np.sum(eva_gt_mask_bin_np>0))

                        # box level mAP
                        for bthi in range(eva_bth_x.shape[0]):
                            bth_eva_small_boxes = eva_small_boxes[bthi]
                            if(isinstance(bth_eva_small_boxes,type(None)) or bth_eva_small_boxes.shape[0]==0):
                                continue
                            for smbx in bth_eva_small_boxes:
                                sub_gt_mask_bin_np,M,blmask = cv_crop_image_by_polygon(eva_gt_mask_bin_np[bthi],smbx,return_mask=True)
                                if(DEF_BOOL_TRAIN_MASK and DEF_BOOL_TRAIN_TXT_MASK):
                                    sub_pred_mask_bin_np = cv2.warpPerspective(eva_pred_mask_bin_np[bthi], M, (blmask.shape[1], blmask.shape[0]))
                                    sub_pred_mask_bin_np[~blmask] = 0
                                    mask_box_map_list.append(np.sum(sub_pred_mask_bin_np==sub_gt_mask_bin_np)/sub_pred_mask_bin_np.size)
                                if(DEF_BOOL_TRAIN_CE and DEF_CE_CH>3):
                                    sub_labels = cv2.warpPerspective(labels[bthi], M, (blmask.shape[1], blmask.shape[0]))
                                    sub_labels[~blmask] = 0
                                    mask_box_ce_map_list.append(np.sum(sub_labels==sub_gt_mask_bin_np)/sub_labels.size)

                    # main detection evaluation
                    for bthi in range(eva_bth_x.shape[0]):
                        eva_image = eva_bth_x[bthi].numpy().astype(np.uint8)
                        precision,recall,fscore=0.0,0.0,0.0
                        bth_eva_boxes = eva_boxes[bthi]
                        if('text' in sample and '#' in sample['text'][bthi]):
                            rmid = [i for i,o in enumerate(sample['text'][bthi]) if(o=='#')]
                            bgbxs = np.array([o for i,o in enumerate(bth_eva_boxes) if(i in rmid)])
                            fgbxs = np.array([o for i,o in enumerate(bth_eva_boxes) if(i not in rmid)])
                            sml_bgbxs = np_box_resize(bgbxs,eva_image.shape[:-1],eva_pred_np.shape[-2:],'polyxy')
                        else:
                            bgbxs = np.array([])
                            fgbxs = bth_eva_boxes
                        if(DEF_BOOL_TRAIN_MASK):
                            eva_region_np = eva_pred_np[bthi,DEF_START_MASK_CH+2]
                            eva_region_np[eva_region_np<0.2]=0
                            # original resolution
                            det_boxes, label_mask, label_list = cv_get_box_from_mask(eva_region_np,region_mean_split=True)
                            if(det_boxes.shape[0]>0):
                                if(log_images_max_cnt>0 and rank==0):
                                    log_images['region_mask_boxed_image']=cv_draw_poly(log_images['image'],det_boxes)
                                det_boxes = np_box_resize(det_boxes,eva_pred_np.shape[-2:],eva_image.shape[:-1],'polyxy')
                                ids,mask_precision,mask_recall = cv_box_match(det_boxes,fgbxs,bgbxs,ovth=ovlap_th)
                                mask_fscore = mask_precision*mask_recall/(mask_precision+mask_recall) if(mask_precision+mask_recall>0)else 0.0
                                
                                if(mask_fscore>fscore):
                                    precision=mask_precision
                                    recall=mask_recall
                            else:
                                mask_precision,mask_recall,mask_fscore=0.0,0.0,0.0

                            # 2x resolution
                            eva_region_2x_np = eva_pred_2x_np[bthi,DEF_START_MASK_CH+2]
                            eva_region_2x_np[eva_region_2x_np<0.2]=0
                            det_boxes_2x, _, _ = cv_get_box_from_mask(eva_region_2x_np,region_mean_split=True)
                            if(det_boxes_2x.shape[0]>0):
                                det_boxes_2x = np_box_resize(det_boxes_2x,eva_pred_2x_np.shape[-2:],eva_image.shape[:-1],'polyxy')
                                _,precision_2x,reacll_2x = cv_box_match(det_boxes_2x,fgbxs,bgbxs,ovth=ovlap_th)
                                recall_list_2x.append(reacll_2x)
                                precision_list_2x.append(precision_2x)
                            else:
                                recall_list_2x.append(0.0)
                                precision_list_2x.append(0.0)

                            # 4x resolution
                            eva_region_4x_np = eva_pred_4x_np[bthi,DEF_START_MASK_CH+2]
                            eva_region_4x_np[eva_region_4x_np<0.2]=0
                            det_boxes_4x, _, _ = cv_get_box_from_mask(eva_region_4x_np,region_mean_split=True)
                            if(det_boxes_4x.shape[0]>0):
                                det_boxes_4x = np_box_resize(det_boxes_4x,eva_pred_4x_np.shape[-2:],eva_image.shape[:-1],'polyxy')
                                _,precision_4x,reacll_4x = cv_box_match(det_boxes_4x,fgbxs,bgbxs,ovth=ovlap_th)
                                recall_list_4x.append(reacll_4x)
                                precision_list_4x.append(precision_4x)
                            else:
                                recall_list_4x.append(0.0)
                                precision_list_4x.append(0.0)

                        if(DEF_BOOL_TRAIN_CE):
                            ce_map = eva_bth_bin_ce_map[bthi].astype(np.float32)
                            ce_det_boxes, ce_label_mask, ce_label_list = cv_get_box_from_mask(ce_map)
                            if(ce_det_boxes.shape[0]>0):
                                if(log_images_max_cnt>0 and rank==0):
                                    log_images['ce_mask_boxed_image']=cv_draw_poly(log_images['image'],ce_det_boxes)
                                ce_det_boxes = np_box_resize(ce_det_boxes,eva_pred_np.shape[-2:],eva_image.shape[:-1],'polyxy')
                                ce_ids,ce_precision,ce_recall = cv_box_match(ce_det_boxes,fgbxs,bgbxs,ovth=ovlap_th)
                                ce_fscore = ce_precision*ce_recall/(ce_precision+ce_recall) if(ce_precision+ce_recall>0)else 0.0
                                if(ce_fscore>fscore):
                                    precision=ce_precision
                                    recall=ce_recall
                            else:
                                ce_precision,ce_recall,ce_fscore=0.0,0.0,0.0

                        if(DEF_BOOL_TRAIN_BOX):
                            image_size_xy = np.array((eva_xnor.shape[-1],eva_xnor.shape[-2]))
                            eva_box_onehit = eva_pred_np[bthi,DEF_START_BOX_CH:DEF_START_BOX_CH+DEF_BOX_HITMAP_CH]
                            if(DEF_BOX_HITMAP_CH==1):
                                eva_box_onehit_yx = np.where(eva_box_onehit[0]>0)
                            else:
                                eva_box_onehit_yx = np.where(eva_box_onehit[1]>eva_box_onehit[0])
                            eva_box_dets = eva_pred_np[bthi,DEF_START_BOX_CH+DEF_BOX_HITMAP_CH:DEF_START_BOX_CH+DEF_BOX_HITMAP_CH+DEF_BOX_CH]
                            eva_box_dets = np.moveaxis(eva_box_dets,0,-1)
                            
                            pred_box_list = []
                            for cy,cx in zip(*eva_box_onehit_yx):
                                pred_cx,pred_cy = eva_box_dets[cy,cx,0:2]*image_size_xy
                                pred_cx+=cx+1
                                pred_cy+=cy+1
                                pred_w,pred_h = eva_box_dets[cy,cx,2:4]*image_size_xy
                                pred_box_list.append(np.array([
                                    [pred_cx-pred_w/2,pred_cy-pred_h/2],
                                    [pred_cx+pred_w/2,pred_cy-pred_h/2],
                                    [pred_cx+pred_w/2,pred_cy+pred_h/2],
                                    [pred_cx-pred_w/2,pred_cy+pred_h/2],
                                    ]))
                            pred_box_list = np.array(pred_box_list)

                            box_ids,box_precision,box_recall = cv_box_match(pred_box_list,fgbxs,bgbxs,ovth=ovlap_th)
                            box_fscore = box_precision*box_recall/(box_precision+box_recall) if(box_precision+box_recall>0)else 0.0
                            if(box_fscore>fscore):
                                precision=box_precision
                                recall=box_recall

                        recall_list.append(recall)
                        precision_list.append(precision)
                    
                    if(logger and rank==0 and stepi in log_eval_id):
                        bximg = eva_image
                        bximg = cv_draw_poly(bximg,eva_boxes[-1],text='GT',color=(0,255,0))
                        if(DEF_BOOL_TRAIN_MASK and det_boxes.shape[0]>0):
                            bximg = cv_draw_poly(bximg,det_boxes,text='Pmk',color=(255,0,0))
                        if(DEF_BOOL_TRAIN_CE and ce_det_boxes.shape[0]>0):
                            bximg = cv_draw_poly(bximg,ce_det_boxes,text='Pce',color=(0,0,255))
                        if(DEF_BOOL_TRAIN_BOX and pred_box_list.shape[0]>0):
                            bximg = cv_draw_poly(bximg,pred_box_list,text='Pbox',color=(255,128,255))
                        imgs = [bximg]
                        if(DEF_BOOL_TRAIN_MASK):
                            imgs.append(cv_heatmap(eva_pred_np[-1,DEF_START_MASK_CH+2]))
                        if(DEF_BOOL_TRAIN_CE):
                            bince = eva_bth_bin_ce_map[-1]*255
                            imgs.append(np.stack([bince,bince,bince],-1))
                        logger.add_image('Eval/epoch {}'.format(epoch), concatenate_images(imgs), eval_img_log_cnt,dataformats='HWC')
                        logger.flush()
                        eval_img_log_cnt+=1
                    if(log_images_max_cnt>0 and rank==0):
                        tmp = sample['name'][0]
                        for o in log_images:
                            save_image(os.path.join(work_dir,
                                time_start.strftime("%Y%m%d-%H%M%S"),'eval_imgs',
                                "epoch{:03d}".format(epoch+1),
                                "{}_{}_ep{}".format(o,tmp,epoch+1)),log_images[o])
                        log_images_max_cnt-=1
                # end of eval step
                recall_np = np.mean(np.array(recall_list,dtype=np.float32))
                recall_gpu = torch.tensor(recall_np).cuda(non_blocking=True)
                precision_np = np.mean(np.array(precision_list,dtype=np.float32))
                precision_gpu = torch.tensor(precision_np).cuda(non_blocking=True)
                if(world_size>1):
                    dist.all_reduce(recall_gpu)
                    dist.all_reduce(precision_gpu)
                    
                recall_np = recall_gpu.item()/world_size
                precision_np = precision_gpu.item()/world_size
                f_np = 2*precision_np*recall_np/(precision_np+recall_np) if(precision_np+recall_np>0)else 0.0
                all_recall.append(recall_np)
                all_precision.append(precision_np)

                # if(last_max_recall<recall_np):
                #     if(last_max_recall!=0):
                #         bool_hit_max=True
                #     last_max_recall = recall_np
                if(last_max_precision<precision_np):
                    if(last_max_precision!=0):
                        bool_hit_max=True
                    last_max_precision = precision_np
                # if(last_max_fscore<f_np):
                #     if(last_max_fscore!=0):
                #         bool_hit_max=True
                #     last_max_fscore = f_np

                if(logger and rank==0):
                    logger.add_scalar('Eval/recall', recall_np, epoch)
                    logger.add_scalar('Eval/precision', precision_np, epoch)
                    logger.add_scalar('Eval/F-score',f_np, epoch)
                    if(DEF_BOOL_TRAIN_MASK):
                        r2x = np.mean(np.array(recall_list_2x,dtype=np.float32))
                        p2x = np.mean(np.array(precision_list_2x,dtype=np.float32))
                        f2x = 2*r2x*p2x/(r2x+p2x) if(r2x+p2x>0)else 0.0
                        logger.add_scalar('Eval/recall_2x', r2x, epoch)
                        logger.add_scalar('Eval/precision_2x', p2x, epoch)
                        logger.add_scalar('Eval/F-score_2x',f2x, epoch)
                        r2x = np.mean(np.array(recall_list_4x,dtype=np.float32))
                        p2x = np.mean(np.array(precision_list_4x,dtype=np.float32))
                        f2x = 2*r2x*p2x/(r2x+p2x) if(r2x+p2x>0)else 0.0
                        logger.add_scalar('Eval/recall_4x', r2x, epoch)
                        logger.add_scalar('Eval/precision_4x', p2x, epoch)
                        logger.add_scalar('Eval/F-score_4x',f2x, epoch)
                    if(mask_loss_list):
                        logger.add_scalar('Eval/global text map',np.mean(np.stack(mask_loss_list)), epoch)
                    if(mask_ce_loss_list):
                        logger.add_scalar('Eval/global ce text map',np.mean(np.stack(mask_ce_loss_list)), epoch)
                    if(mask_box_map_list):
                        logger.add_scalar('Eval/box text map',np.mean(np.stack(mask_box_map_list)), epoch)
                    if(mask_box_ce_map_list):
                        logger.add_scalar('Eval/box ce text map',np.mean(np.stack(mask_box_ce_map_list)), epoch)
                    logger.flush()
                
                if(last_max_fscore>0.4 and (last_max_fscore-f_np)>0.5*last_max_fscore):
                    bool_low_fscore=True

                if(args.save and bool_hit_max and rank==0):
                    fmdir,fmname = os.path.split(args.save)
                    fmname = 'max_eval_'+fmname
                    finalname = os.path.join(fmdir,fmname)
                    print("Saving model at {}...".format(finalname))
                    torch.save(model.state_dict(),finalname)

                if((epoch+1)%10==0):
                    last_recall = all_recall[-10:]
                    last_precision = all_precision[-10:]
                    if(last_max_recall>0.4 and last_max_precision>0.4 and last_max_recall not in last_recall and last_max_precision not in last_precision):
                        last_recall_mean = np.mean(np.array(last_recall))
                        last_precision_mean = np.mean(np.array(last_precision))
                        if((last_max_recall-last_recall_mean)<0.2*last_max_recall or (last_max_precision-last_precision_mean)<0.2*last_max_precision):
                            for param_group in optimizer.param_groups:
                                curlr = param_group['lr']
                                break
                            print("Adjust learning rate {}->{}.".format(curlr,curlr*0.9))
                            adjust_learning_rate(optimizer,0.9)


        if(logger and rank==0):
            log_i,box_num = 0,boxes[0].shape[0]
            for i,o in enumerate(boxes):
                if(box_num<o.shape[0]):
                    log_i=i
                    box_num = o.shape[0]
            if(DEF_BOOL_TRAIN_MASK):
                pred_regi = pred[log_i,DEF_START_MASK_CH+2].to('cpu').detach().numpy()
                smx = cv2.resize(x[log_i].numpy().astype(np.uint8),(pred.shape[-1],pred.shape[-2]))
                smx = cv_mask_image(smx,cv_heatmap(pred_regi))
                logimg = [smx]
                if(DEF_BOOL_TRAIN_TXT_MASK):
                    pred_mask = pred[log_i,DEF_START_MASK_CH+0].to('cpu').detach().numpy()
                    pred_mask = (pred_mask*255.0).astype(np.uint8)
                    pred_mask = np.stack([pred_mask,pred_mask,pred_mask],-1)
                    pred_edge = pred[log_i,DEF_START_MASK_CH+1].to('cpu').detach().numpy()
                    pred_edge = (pred_edge*255.0).astype(np.uint8)
                    pred_edge = np.stack([pred_edge,pred_edge,pred_edge],-1)
                    logimg.append(pred_mask)
                    logimg.append(pred_edge)

                if(DEF_MASK_CH>3):
                    # text threshold regression
                    pred_threshold = pred[log_i,DEF_START_MASK_CH+3].to('cpu').detach().numpy()
                    pred_threshold = (pred_threshold*255).astype(np.uint8)
                    pred_threshold = np.stack([pred_threshold,pred_threshold,pred_threshold],-1)
                    logimg.append(pred_threshold)

                logger.add_image('Prediction', concatenate_images(logimg), epoch,dataformats='HWC')
                if(DEF_BOOL_TRAIN_TXT_MASK and (b_have_mask or args.genmask)):
                    gt_mask = np.stack([mask_np[log_i],mask_np[log_i],mask_np[log_i]],-1)
                    gt_edge = np.stack([edge_np[log_i],edge_np[log_i],edge_np[log_i]],-1)
                    gt_regi = cv_heatmap(region_mask_np[log_i,:,:,0])
                    smx = cv2.resize(x[log_i].numpy().astype(np.uint8),(gt_mask.shape[1],gt_mask.shape[0]))
                    smx = cv_mask_image(smx,gt_regi)
                    logimg = [smx,gt_mask,gt_edge]
                    if(DEF_MASK_CH>3):
                        gt_threshold = threshold_np[log_i]
                        if(len(gt_threshold.shape)==2):
                            gt_threshold = np.expand_dims(gt_threshold,-1)
                        if(gt_threshold.shape[-1]==1):
                            gt_threshold = np.concatenate([gt_threshold,gt_threshold,gt_threshold],-1)
                        logimg.append(gt_threshold)

                    logger.add_image('GT', concatenate_images(logimg), epoch,dataformats='HWC')

                if(DEF_BOOL_LOG_LEVEL_MASK):
                    try:
                        b1_mask = feat.b1_mask[log_i].to('cpu').permute(1,2,0).detach().numpy()
                        b2_mask = feat.b2_mask[log_i].to('cpu').permute(1,2,0).detach().numpy()
                        b3_mask = feat.b3_mask[log_i].to('cpu').permute(1,2,0).detach().numpy()
                        b4_mask = feat.b4_mask[log_i].to('cpu').permute(1,2,0).detach().numpy()

                        b2_mask = cv2.resize(b2_mask,(b1_mask.shape[1],b1_mask.shape[0]))
                        b3_mask = cv2.resize(b3_mask,(b1_mask.shape[1],b1_mask.shape[0]))
                        b4_mask = cv2.resize(b4_mask,(b1_mask.shape[1],b1_mask.shape[0]))
                        line = np.ones((b1_mask.shape[0],3,3),dtype=np.uint8)*255
                        img = np.concatenate((
                            cv_heatmap(b3_mask[:,:,0]),line,
                            cv_heatmap(b4_mask[:,:,0])),-2)
                        logger.add_image('B3|B4 BG', img, epoch,dataformats='HWC')
                        img = np.concatenate((
                            cv_heatmap(b3_mask[:,:,1]),line,
                            cv_heatmap(b4_mask[:,:,1])),-2)
                        logger.add_image('B3|B4 region', img, epoch,dataformats='HWC')
                        img = np.concatenate((
                            cv_heatmap(b1_mask[:,:,1]),line,
                            cv_heatmap(b2_mask[:,:,1])),-2)
                        logger.add_image('B1|B2 edge', img, epoch,dataformats='HWC')
                        img = np.concatenate((
                            cv_heatmap(b1_mask[:,:,0]),line,
                            cv_heatmap(b2_mask[:,:,0]),line,
                            cv_heatmap(b3_mask[:,:,2]),line,
                            cv_heatmap(b4_mask[:,:,2])),-2)
                        logger.add_image('B1|B2|B3|B4 txt', img, epoch,dataformats='HWC')
                    except Exception as e:
                        print(str(e))

            if(DEF_BOOL_TRAIN_CE):
                labels = torch.argmax(pred[log_i:log_i+1,DEF_START_CE_CH:DEF_START_CE_CH+3],dim=1)
                labels = labels[0].cpu().detach().numpy()
                gtlabels = ce_y[log_i]
                labels = cv_labelmap(labels)
                gtlabels = cv_labelmap(gtlabels)
                img = cv2.resize(x[log_i].numpy().astype(np.uint8),(gtlabels.shape[1],gtlabels.shape[0]))
                logger.add_image('Region labels Image|GT|Pred', concatenate_images([img,gtlabels,labels]), epoch,dataformats='HWC')
                if(DEF_BOOL_TRAIN_TXT_MASK and DEF_CE_CH>3):
                    t='Text labels Image'
                    img_lst = [img]
                    if(b_have_mask or args.genmask):
                        t+='|GT'
                        gtlabels = (mask_np[log_i]>0).astype(np.uint8)
                        img_lst.append(cv_labelmap(gtlabels))
                    t+='|Pred'
                    labels = torch.argmax(pred[log_i:log_i+1,DEF_START_CE_CH+3:DEF_START_CE_CH+DEF_CE_CH],dim=1)
                    labels = labels[0].cpu().detach().numpy() 
                    img_lst.append(cv_labelmap(labels))
                    logger.add_image(t, concatenate_images(img_lst), epoch,dataformats='HWC')

            if(DEF_BOOL_TRAIN_BOX):
                pred_box = pred[log_i,DEF_START_BOX_CH+DEF_BOX_HITMAP_CH:DEF_START_BOX_CH+DEF_BOX_CH+DEF_BOX_HITMAP_CH].to('cpu').detach().numpy()
                div_num = DEF_POLY_NUM//2
                smx = cv2.resize(x[log_i].numpy().astype(np.uint8),(pred_box.shape[-1],pred_box.shape[-2]))
                small_image_size_xy = np.array([pred_box.shape[-1],pred_box.shape[-2]])
                batch_boxes = boxes[log_i]
                boxes_small = np_box_resize(batch_boxes,x.shape[1:3],pred_box.shape[-2:],'polyxy')
                ct_box = np_polybox_center(boxes_small)
                if(DEF_BOOL_POLY_REGRESSION):
                    poly_gt_list = []
                    c_pred_box_list = []
                    for boxi in range(boxes_small.shape[0]):
                        poly_gt = np_split_polygon(boxes_small[boxi],div_num)
                        cx,cy = ct_box[boxi].astype(np.uint16)
                        if(cx>=pred_box.shape[2] or cy>=pred_box.shape[1]):
                            continue
                        c_pred_box = pred_box[:,cy-1,cx-1].reshape(-1,2)*small_image_size_xy+ct_box[boxi]
                        poly_gt_list.append(poly_gt)
                        c_pred_box_list.append(c_pred_box)
                    gt_img = cv_draw_poly(smx,poly_gt_list,color=(0,255,0),point_emphasis=True)
                    pred_img = cv_draw_poly(smx,c_pred_box_list,color=(0,0,255),point_emphasis=True)
                else:
                    c_pred_box_list = []
                    for boxi in range(boxes_small.shape[0]):
                        cx,cy = ct_box[boxi].astype(np.uint16)
                        if(cx>=pred_box.shape[2] or cy>=pred_box.shape[1]):
                            continue
                        pred_cx,pred_cy = pred_box[0:2,cy-1,cx-1]*small_image_size_xy+ct_box[boxi]
                        pred_w,pred_h = pred_box[2:4,cy-1,cx-1]*small_image_size_xy
                        c_pred_box_list.append(np.array([
                            [pred_cx-pred_w/2,pred_cy-pred_h/2],
                            [pred_cx+pred_w/2,pred_cy-pred_h/2],
                            [pred_cx+pred_w/2,pred_cy+pred_h/2],
                            [pred_cx-pred_w/2,pred_cy+pred_h/2],
                            ]))
                    gt_img = cv_draw_poly(smx,boxes_small,color=(0,255,0),point_emphasis=True)
                    pred_img = cv_draw_poly(smx,c_pred_box_list,color=(0,0,255),point_emphasis=True)
                logger.add_image('BOX GT|Pred', concatenate_images([gt_img,pred_img]), epoch,dataformats='HWC')

            if(DEF_BOOL_TRACKING):
                img = batch_x_np
                img[:sub_img.shape[0],:sub_img.shape[1]] = sub_img
                match_map_np = match_map[0,0].to('cpu').detach().numpy()
                track_gt = cv_mask_image(img,cv_heatmap(sub_ch_mask))
                track_pred = cv_mask_image(img,cv_heatmap(match_map_np))
                if(track_pred.shape!=track_gt.shape):
                    track_gt = cv2.resize(track_gt,(track_pred.shape[1],track_pred.shape[0]))
                line = np.ones((track_pred.shape[0],3,3),dtype=np.uint8)*255
                img = np.concatenate((track_gt,line,track_pred),-2)
                logger.add_image('Tracking GT|Pred', img, epoch,dataformats='HWC')
                
            logger.flush()

        if(rank == 0):
            time_usage = datetime.now() - time_c
            time_c = datetime.now()
            log.write('Epoch [{}/{}], Loss: {:.4f}\n'.format(epoch + 1, args.epoch,loss.item()))
            if(args.eval):
                log.write('\tRecall: {:.4f},Precision: {:.4f}, F-score: {:.4f}\n'.format(recall_np, precision_np,f_np))
            try:                         
                log.write("Time usage: {} Day {} Second.\n\n".format(time_usage.days,time_usage.seconds))
            except:
                log.write("Time usage: {} Day {} Second.\n\n".format(time_usage.day,time_usage.second))
            log.flush()
        if(args.eval and bool_low_fscore):
            break

        if(args.save and rank == 0):
            # model_cpu = model.cpu()
            print("Saving model at {}...".format(args.save))
            torch.save(model.state_dict(),args.save)
        # End of for

    if(rank == 0):
        time_usage = datetime.now() - time_start
        try:
            log.write("Total time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
        except:
            log.write("Total time usage: {} Day {} Second.\n".format(time_usage.day,time_usage.second))
        log.close()
    return 0

def init_process(rank, world_size, fn, args):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    print("Process id {}/{} init.".format(rank,world_size))
    fn(rank,world_size,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
    #                     help='number of data loading workers (default: 4)')
    # parser.add_argument('-g', '--gpus', default=1, type=int,
    #                     help='number of gpus per node')
    # parser.add_argument('-nr', '--nr', default=0, type=int,
    #                     help='ranking within the nodes')
    # parser.add_argument('--epochs', default=2, type=int, metavar='N',
    #                     help='number of total epochs to run')

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

        if(args.multi_gpu and gpus>1):
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '8888'
            mp.spawn(init_process, nprocs=gpus, args=(gpus,train,args,))
        else:
            train(0, 1, args)
        last_save = args.save