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
from lib.model.pixel_map import PIX_Unet
from lib.model.siamfc import SiameseNet
from lib.loss.mseloss import *
from lib.loss.logits import FocalLoss
from lib.dataloader.total import Total
from lib.dataloader.icdar import *
from lib.dataloader.msra import MSRA
from lib.dataloader.icdar_video import ICDARV
from lib.dataloader.minetto import Minetto
from lib.dataloader.base import split_dataset_cls_to_train_eval
from lib.dataloader.synthtext import SynthText
from lib.utils.img_hlp import *
from lib.utils.log_hlp import *
from lib.utils.net_hlp import adjust_learning_rate
from lib.config.train_default import cfg as tcfg
from dirs import *

DEF_MAX_TRACKING_BOX_NUMBER = 2
DEF_MIN_INPUT_SIZE = 42

def train(rank, world_size, args):

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
    if(rank == 0):
        log = sys.stdout
    args.task = args.task.lower()
    DEF_TRAIN_MASK = 'mask' in args.task
    DEF_TRAIN_TEXT = 'text' in args.task

    batch_size = args.batch
    torch.manual_seed(0)
    torch.cuda.set_device(rank)

    basenet = PIX_Unet(mask_ch=2,min_mask_ch=32,include_final=args.have_fc,basenet=args.basenet,min_upc_ch=128,pretrained=True).float()
    model = SiameseNet(basenet)
    if(args.load and os.path.exists(args.load)):
        if(rank==0):
            log.write("Loading parameters at {}.\n".format(args.load))
        model.load_state_dict(copyStateDict(torch.load(args.load)))
    model = model.cuda(rank)

    if(world_size>1):
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion_mask = MASK_MSE_LOSS().cuda(rank)
    criterion_logit = FocalLoss().cuda(rank)
    criterion = MSE_2d_Loss(pixel_sum=False).cuda(rank)

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
    
    args.dataset = args.dataset.lower()
    if(args.dataset=="ttt"):
        train_dataset = Total(
            os.path.join(DEF_TTT_DIR,'images','train'),
            os.path.join(DEF_TTT_DIR,'gt_pixel','train'),
            os.path.join(DEF_TTT_DIR,'gt_txt','train'),
            image_size=image_size,
            include_bg=True,
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
            bth_image_np = x.numpy()
            b_have_mask = bool('mask' in sample)
            if(b_have_mask):
                bth_text_mask = sample['mask']
                bth_text_mask_np = bth_text_mask.numpy()

            bth_boxes = sample['box']
            # data argumentation
            if(random_b):
                boxest_list = []
                xt_list = []
                if(b_have_mask):
                    text_maskt_list = []
                
                for i in range(bth_image_np.shape[0]):
                    if(not isinstance(bth_boxes[i],type(None)) and bth_boxes[i].shape[0]>0):
                        imgt,boxt,M = cv_random_image_process(bth_image_np[i],bth_boxes[i],crop_to_box=not('text' in sample and '#' in sample['text'][i]))
                    else:
                        imgt,boxt,M = cv_random_image_process(bth_image_np[i],None,crop_to_box=False)

                    xt_list.append(imgt)
                    boxest_list.append(boxt if(not isinstance(boxt,type(None)))else bth_boxes[i])
                    if(b_have_mask):
                        text_maskt_list.append(cv2.warpAffine(bth_text_mask_np[i], M[:-1], (bth_image_np.shape[2],bth_image_np.shape[1])))

                bth_boxes = boxest_list
                bth_image_np = np.stack(xt_list)
                x = torch.from_numpy(bth_image_np)
                if(b_have_mask):
                    bth_text_mask_np = np.stack(text_maskt_list).astype(np.uint8)
                    if(len(bth_text_mask_np.shape)==3):
                        bth_text_mask_np = np.expand_dims(bth_text_mask_np,-1)
                    bth_text_mask = torch.from_numpy(bth_text_mask_np)
            
            # generate region mask
            bth_region_mask_list = []
            bth_binary_region_mask_list = []
            for o in bth_boxes:
                if(o.shape[0]==0):
                    bth_region_mask_list.append(np.zeros(x.shape[-3:-1],dtype=np.float32))
                    bth_binary_region_mask_list.append(np.zeros(x.shape[-3:-1],dtype=np.uint8))
                else:
                    fmask,bmask = cv_gen_gaussian_by_poly(o,x.shape[-3:-1],return_mask=True)
                    bth_region_mask_list.append(fmask)
                    bth_binary_region_mask_list.append(bmask)
            
            bth_region_mask_np = np.array(bth_region_mask_list,dtype=np.float32)
            bth_region_mask = torch.from_numpy(bth_region_mask_np)
            # bth_region_mask_np = np.expand_dims(bth_region_mask_np,-1)
            bth_binary_region_mask_np = np.array(bth_binary_region_mask_list,dtype=np.uint8)
            # bth_binary_region_mask_np = np.expand_dims(bth_binary_region_mask_np,-1)

            # generate selection mask, 0 for ignored, 1 for background, 2 for positive
            bth_region_selection_list = []
            if(not isinstance(bth_boxes,type(None))):
                for i in range(bth_image_np.shape[0]):
                    if('text' in sample and '#' in sample['text'][i]):
                        selection = cv_gen_binary_mask_by_poly(bth_boxes[i],bth_image_np.shape[-3:-1],
                            default_value=1,default_fill=2,
                            box_fill_value_list=[2 if(o!='#')else 0 for o in sample['text'][i]]
                            )
                    elif(not isinstance(bth_boxes[i],type(None))):
                        selection = cv_gen_binary_mask_by_poly(bth_boxes[i],bth_image_np.shape[-3:-1],
                            default_value=1,default_fill=2)
                    else:
                        selection = np.ones(bth_image_np.shape[-3:-1],dtype=np.uint8)
                    bth_region_selection_list.append(selection)
                bth_region_selection_np = np.array(bth_region_selection_list,dtype=np.uint8)
            else:
                bth_region_selection_np = np.ones(bth_image_np.shape[:-1],dtype=np.uint8)
            bth_region_selection = torch.from_numpy(bth_region_selection_np)
            
            # data normolization
            xnor = x.float().cuda(non_blocking=True)
            xnor = torch_img_normalize(xnor).permute(0,3,1,2)
            
            # Forward pass
            pred,feat = model(xnor)
            # Loss
            loss_dict = {}

            loss_dict['region_mask_regression']=criterion_mask(pred[:,0],bth_region_mask.cuda(non_blocking=True),bth_region_selection.cuda(non_blocking=True))

            if(DEF_TRAIN_TEXT and b_have_mask):
                # text mask regression
                # find non-text positive region and ignore it
                # generate selection mask, 0 for ignored, 1 for background, 2 for positive
                # bth_text_selection_list = []
                # if(not isinstance(bth_boxes,type(None))):
                #     for i in range(bth_image_np.shape[0]):
                #         if(not isinstance(bth_boxes[i],type(None))):
                #             box_id = []
                #             for j,o in enumerate(bth_boxes[i]):
                #                 sub_text_mask, M, sub_region_mask = cv_crop_image_by_polygon(bth_text_mask_np[i],o,return_mask=True)
                #                 text_post_num = np.sum(sub_text_mask>0)
                #                 reg_post_num = np.sum(sub_region_mask>0)
                #                 if(text_post_num>int(0.2*reg_post_num) and text_post_num<int(0.9*reg_post_num)):
                #                     box_id.append(2)
                #                 else:
                #                     box_id.append(0)

                #             selection = cv_gen_binary_mask_by_poly(bth_boxes[i],bth_image_np.shape[-3:-1],
                #                 default_value=1, default_fill=2, box_fill_value_list=box_id)
                #         else:
                #             selection = np.ones(bth_image_np.shape[-3:-1],dtype=np.uint8)
                #         bth_text_selection_list.append(selection)
                #     bth_text_selection_np = np.array(bth_text_selection_list,dtype=np.uint8)
                # else:
                #     bth_text_selection_np = np.ones(bth_image_np.shape[:-1],dtype=np.uint8)
                # bth_text_selection = torch.from_numpy(bth_text_selection_np)
                bth_text_selection = bth_region_selection
                loss_dict['region_text_regression']=criterion_mask(pred[:,1],bth_text_mask.float().cuda(non_blocking=True).permute(0,3,1,2)/255,bth_text_selection.cuda(non_blocking=True))

            # tracking
            tracking_loss_list = []
            tracking_loger = []
            for i in range(bth_image_np.shape[0]):
                if(isinstance(bth_boxes,type(None)) or isinstance(bth_boxes[i],type(None))):
                    continue
                loc_loger = []
                # add random affine transform
                imgt,boxt,M = cv_random_image_process(bth_image_np[i],bth_boxes[i],crop_to_box=False)
                rect_boxt = np_polybox_minrect(boxt,'polyxy')
                ws = np.linalg.norm(rect_boxt[:,0]-rect_boxt[:,1],axis=-1)
                hs = np.linalg.norm(rect_boxt[:,0]-rect_boxt[:,3],axis=-1)
                inds = np.argsort(ws*hs)[::-1]
                if(args.crop):
                    slc_recbox = boxt[inds[:DEF_MAX_TRACKING_BOX_NUMBER]]
                    for bxi,curbx in enumerate(slc_recbox):
                        simg,Mcrop = cv_crop_image_by_polygon(imgt,curbx,w_min=DEF_MIN_INPUT_SIZE,h_min=DEF_MIN_INPUT_SIZE)
                        track_x = torch.from_numpy(np.expand_dims(simg,0)).float().cuda(non_blocking=True)
                        track_xnor = torch_img_normalize(track_x).permute(0,3,1,2)
                        track_pred,track_feat = model(track_xnor)
                        slc_feat = track_feat.upb0
                        if(args.fresize):
                            slc_feat = F.interpolate(slc_feat,size=(10,10), mode='bilinear', align_corners=False)
                        if(world_size>1):
                            # multi-GPU function
                            score = model.module.match_corr(slc_feat,feat.upb0[i:i+1])
                        else:
                            score = model.match_corr(slc_feat,feat.upb0[i:i+1])
                        target_box = bth_boxes[i][inds[bxi]]
                        target_box = np_box_resize(target_box,bth_image_np.shape[-3:-1],score.shape[2:],'polyxy')
                        if(args.logit):
                            gt_score_np = cv_gen_center_binary_mask_by_poly(target_box,score.shape[2:])
                            gt_score_np = gt_score_np.reshape(1,1,gt_score_np.shape[0],gt_score_np.shape[1])
                            gt_score = torch.from_numpy(gt_score_np).float().cuda(non_blocking=True)
                            tracking_loss_list.append(criterion_logit(score,gt_score))
                            loc_loger.append((boxt[inds[bxi]],torch.sigmoid(score[0,0]).detach().cpu().numpy()))
                        else:
                            gt_score_np = cv_gen_gaussian_by_poly(target_box,score.shape[2:])
                            gt_score = torch.from_numpy(np.expand_dims(gt_score_np,0)).float().cuda(non_blocking=True)
                            tracking_loss_list.append(criterion(score,gt_score))
                            loc_loger.append((boxt[inds[bxi]],gt_score_np,score[0,0].detach().cpu().numpy()))

                else:
                    slc_recbox = rect_boxt[inds[:DEF_MAX_TRACKING_BOX_NUMBER]]
                    # torch tensor
                    track_x = torch.from_numpy(np.expand_dims(imgt,0)).float().cuda(non_blocking=True)
                    track_xnor = torch_img_normalize(track_x).permute(0,3,1,2)
                    # run
                    track_pred,track_feat = model(track_xnor)
                    sml_slc_recbox = np_box_resize(slc_recbox,bth_image_np.shape[-3:-1],track_pred.shape[-2:],'polyxy')
                    
                    # search_pred, search_feat = model(xnor[i:i+1])
                    for bxi,smbx_rect in enumerate(sml_slc_recbox):
                        try:
                        # if(1):
                            # crop
                            x1,y1 = smbx_rect[0].astype(np.uint8)
                            x2,y2 = smbx_rect[2].astype(np.uint8)
                            if(y1>y2):
                                y1,y2=y2,y1
                            if(x1>x2):
                                x1,x2=x2,x1
                            x1,y1=max(x1,0),max(y1,0)
                            slc_feat = track_feat.upb0[:,:,y1:y1+abs(y2-y1),x1:x1+abs(x2-x1)]
                            if(args.fresize):
                                slc_feat = F.interpolate(slc_feat,size=(10,10), mode='bilinear', align_corners=False)
                            if(world_size>1):
                                # multi-GPU function
                                score = model.module.match_corr(slc_feat,feat.upb0[i:i+1])
                            else:
                                score = model.match_corr(slc_feat,feat.upb0[i:i+1])
                            # score = score/torch.max(score)
                            target_box = bth_boxes[i][inds[bxi]]
                            target_box = np_box_resize(target_box,bth_image_np.shape[-3:-1],score.shape[2:],'polyxy')
                            if(args.logit):
                                gt_score_np = cv_gen_center_binary_mask_by_poly(target_box,score.shape[2:])
                                gt_score_np = gt_score_np.reshape(1,1,gt_score_np.shape[0],gt_score_np.shape[1])
                                gt_score = torch.from_numpy(gt_score_np).float().cuda(non_blocking=True)
                                tracking_loss_list.append(criterion_logit(score,gt_score))
                                loc_loger.append((boxt[inds[bxi]],torch.sigmoid(score[0,0]).detach().cpu().numpy()))
                            else:
                                gt_score_np = cv_gen_gaussian_by_poly(target_box,score.shape[2:])
                                gt_score = torch.from_numpy(np.expand_dims(gt_score_np,0)).float().cuda(non_blocking=True)
                                tracking_loss_list.append(criterion(score,gt_score))
                                loc_loger.append((boxt[inds[bxi]],gt_score_np,score[0,0].detach().cpu().numpy()))
                            
                        except Exception as e:
                            if(rank==0):
                                log.write("Err at file: {}, box: {}, err: {}.\n".format(sample['name'][i],smbx_rect,str(e)))
                                log.flush()
                tracking_loger.append([imgt,loc_loger])
            if(tracking_loss_list):
                loss_dict['tracking']=torch.stack(tracking_loss_list).mean()
            else:
                loss_dict['tracking']=torch.zeros_like(loss_dict['region_mask_regression'])

            loss = 0.0
            for keyn,value in loss_dict.items():
                loss+=value
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_list.append(loss.item())

            if(rank==0 and logger):
                logger.add_scalar('Loss/total loss', loss.item(), epoch*total_step+stepi)
                for keyn,value in loss_dict.items():
                    logger.add_scalar('Loss/{}'.format(keyn), value.item(), epoch*total_step+stepi)
                if(args.lr_decay):
                    logger.add_scalar('LR rate', optimizer.param_groups[0]['lr'], epoch*total_step+stepi)
                logger.flush()
            org_images = bth_image_np.astype(np.uint8)
            pred_np = pred.detach().cpu().numpy()
        
        # ==============
        # End of epoch
        # ==============
        if(rank==0):
           
            if(args.save):
                # model_cpu = model.cpu()
                log.write("Saving model at {}...\n".format(args.save))
                log.flush()
                torch.save(model.state_dict(),args.save)

            time_usage = datetime.now() - time_c
            time_c = datetime.now()
            log.write('Epoch [{}/{}], Loss: {:.4f}\n'.format(epoch + 1, args.epoch,loss.item()))
            # if(args.eval):
            #     log.write('\tRecall: {:.4f},Precision: {:.4f}, F-score: {:.4f}\n'.format(recall_np, precision_np,f_np))
            try:                         
                log.write("Time usage: {} Day {} Second.\n\n".format(time_usage.days,time_usage.seconds))
            except:
                log.write("Time usage: {} Day {} Second.\n\n".format(time_usage.day,time_usage.second))
            log.flush()

            if(logger):
                org_images = bth_image_np.astype(np.uint8)
                pred_np = pred.detach().cpu().numpy()
                for i in range(org_images.shape[0]):
                    logger.add_image('Region Prediction: Org|GT|Pred',
                        concatenate_images([org_images[i],cv_heatmap(bth_region_mask_np[i]),cv_heatmap(pred_np[i,0])]),
                        org_images.shape[0]*epoch+i,dataformats='HWC')
                    if(b_have_mask):
                        logger.add_image('Text Prediction: Org|GT|Pred',
                            concatenate_images([org_images[i],bth_text_mask_np[i].astype(np.uint8),(pred_np[i,1]>0).astype(np.uint8)*255]),
                            org_images.shape[0]*epoch+i,dataformats='HWC')
                    
                    track_img,loc_loger = tracking_loger[i]
                    if(args.logit):
                        for j,(boxt,pred_score_np) in enumerate(loc_loger):
                            bx_track_img = cv_draw_poly(track_img,boxt)
                            np_one_hit_mask = (pred_score_np>0.5).astype(np.uint8)*255
                            np_one_hit_mask = np.stack([np_one_hit_mask,np_one_hit_mask,np_one_hit_mask],axis=-1)
                            logger.add_image('Tracking: Org_img|track_img|Pred_sc',
                                concatenate_images([org_images[i],bx_track_img,np_one_hit_mask]),
                                DEF_MAX_TRACKING_BOX_NUMBER*org_images.shape[0]*epoch+i*DEF_MAX_TRACKING_BOX_NUMBER+j,dataformats='HWC')
                    else:
                        for j,(boxt,gt_score_np,pred_score_np) in enumerate(loc_loger):
                            bx_track_img = cv_draw_poly(track_img,boxt)
                            logger.add_image('Tracking: Org_img|track_img|GT_sc|Pred_sc',
                                concatenate_images([org_images[i],bx_track_img,cv_heatmap(gt_score_np),cv_heatmap(pred_score_np)]),
                                DEF_MAX_TRACKING_BOX_NUMBER*org_images.shape[0]*epoch+i*DEF_MAX_TRACKING_BOX_NUMBER+j,dataformats='HWC')
                logger.flush()

    
def init_process(rank, world_size, fn, args):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    print("Process id {}/{} init.".format(rank,world_size))
    fn(rank,world_size,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_gpu', help='Set --multi_gpu to enable multi gpu training.',action="store_true")
    parser.add_argument('--fresize', help='Set --fresize to enable feature resize when tracking.',action="store_true")
    parser.add_argument('--opt', help='PKL path or name of optimizer.',default='adag')
    parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
    parser.add_argument('--eval', help='Set --eval to enable eval.', action="store_true")
    parser.add_argument('--task', help='Tasks, string of mask, text, cls, box.', default='mask')
    parser.add_argument('--basenet', help='Choose base noework.', default='mobile')
    # parser.add_argument('--tracker', type=str,help='Choose tracker.')
    parser.add_argument('--save', type=str, help='Set --save file_dir if want to save network.')
    parser.add_argument('--load', type=str, help='Set --load file_dir if want to load network.')
    parser.add_argument('--name', help='Name of task.')
    parser.add_argument('--dataset', help='Choose dataset: ctw/svt/ttt.', default='ttt')
    parser.add_argument('--batch', type=int, help='Batch size.',default=2)
    parser.add_argument('--learnrate', type=str, help='Learning rate.',default="0.001")
    parser.add_argument('--epoch', type=str, help='Epoch size.',default="10")
    parser.add_argument('--random', type=int, help='Set 1 to enable random change.',default=0)
    # parser.add_argument('--bxrefine', help='Set --bxrefine to enable box refine.', action="store_true")
    # parser.add_argument('--genmask', help='Set --genmask to enable generated mask.', action="store_true")
    parser.add_argument('--have_fc', help='Set --have_fc to include final level.', action="store_true")
    parser.add_argument('--lr_decay', help='Set --lr_decay to enbable learning rate decay.', action="store_true")
    parser.add_argument('--logit', help='Set --logit to use logit loss function.', action="store_true")
    parser.add_argument('--crop', help='Set --crop to use croped image as input when training.', action="store_true")

    args = parser.parse_args()
    args.dataset = args.dataset.lower() if(args.dataset)else args.dataset
    # args.tracker = args.tracker.lower() if(args.tracker)else args.tracker

    # args.debug = True
    # args.logit = True
    # args.batch=2
    # args.load = "/BACKUP/yom_backup/saved_model/mask_only_pxnet.pth"
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
            "Task name: {}\n".format(args.task)+\
            "Task number: {}/{}\n".format(taskid+1,total_tasks)+\
            "Working DIR: {}\n".format(DEF_WORK_DIR)+\
            "Running with: \n"+\
            "\t Epoch size: {},\n\t Batch size: {}.\n".format(args.epoch,args.batch)+\
            "\t Base network: {}.\n".format(args.basenet)+\
            "\t Include final level: {}.\n".format('Yes' if(args.have_fc)else 'No')+\
            "\t Optimizer: {}.\n".format(cur_opt)+\
            "\t Logit: {}.\n".format('Yes' if(args.logit)else 'No')+\
            "\t Crop tracking: {}.\n".format('Yes' if(args.crop)else 'No')+\
            "\t LR decay: {}.\n".format('Yes' if(args.lr_decay)else 'No')+\
            "\t Tracking feature resize: {}.\n".format('Yes' if(args.fresize)else 'No')+\
            "\t Dataset: {}.\n".format(cur_dataset)+\
            "\t Init learning rate: {}.\n".format(cur_learnrate)+\
            "\t Taks name: {}.\n".format(args.name if(args.name)else 'None')+\
            "\t Load network: {}.\n".format(args.load if(args.load)else 'No')+\
            "\t Save network: {}.\n".format(args.save if(args.save)else 'No')+\
            "========\n\n"
            # "\t Box refine: {}.\n".format('Yes' if(args.bxrefine)else 'No')+\
            # "\t Network: {}.\n".format(args.net)+\
            # "\t Generated mask: {}.\n".format('Yes' if(args.genmask)else 'No')+\
            # "\t Train mask: {}.\n".format('Yes' if(DEF_BOOL_TRAIN_MASK)else 'No')+\
            # "\t Train classifier: {}.\n".format('Yes' if(DEF_BOOL_TRAIN_CE)else 'No')+\
        sys.stdout.write(summarize)
        sys.stdout.flush()
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