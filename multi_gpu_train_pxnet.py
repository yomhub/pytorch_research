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
from lib.model.siamfc import SiamesePXT,SiameseMask,conv2d_dw_group,SiameseCRAFT
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
    if(rank==0):
        logger = SummaryWriter(os.path.join(work_dir,time_start.strftime("%Y%m%d-%H%M%S")))
    else:
        logger = None

    batch_size = args.batch
    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    dev = 'cuda:{}'.format(rank)
    
    if(args.net=='vgg_pur_cls'):
        model = VGG_PUR_CLS(include_b0=True,padding=False,pretrained=True).float()
        DEF_BOOL_TRAIN_MASK = False
        DEF_BOOL_TRAIN_CE = True
        DEF_BOOL_LOG_LEVEL_MASK = False
    elif(args.net=='vggunet_pxmask'):
        model = VGGUnet_PXMASK(include_b0=True,padding=False,pretrained=True).float()
        DEF_BOOL_TRAIN_MASK = True
        DEF_BOOL_TRAIN_CE = False
        DEF_BOOL_LOG_LEVEL_MASK = False
    elif(args.net=='vgg_pxmask'):
        model = VGG_PXMASK(include_b0=True,padding=False,pretrained=True).float()
        DEF_BOOL_TRAIN_MASK = True
        DEF_BOOL_TRAIN_CE = False
        DEF_BOOL_LOG_LEVEL_MASK = True
    elif(args.net=='pix_txt'):
        model = PIX_TXT().float()
        DEF_BOOL_TRAIN_MASK = True
        DEF_BOOL_TRAIN_CE = False
        DEF_BOOL_LOG_LEVEL_MASK = True
    else:
        model = PIX_MASK().float()
        DEF_BOOL_TRAIN_MASK = True
        DEF_BOOL_TRAIN_CE = True
        DEF_BOOL_LOG_LEVEL_MASK = True
        
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

    DEF_START_MASK_CH = 0
    DEF_MASK_CH = 3
    DEF_START_CE_CH = DEF_START_MASK_CH+DEF_MASK_CH if(DEF_BOOL_TRAIN_MASK)else 0
    DEF_CE_CH = 3
    DEF_MAX_TRACKING_BOX_NUMBER = 1

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
    # Wrap the model
    if(world_size>1):
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # Data loading code

    if(args.dataset=="ttt"):
        train_dataset = Total(
            os.path.join(DEF_TTT_DIR,'images','train'),
            os.path.join(DEF_TTT_DIR,'gt_pixel','train'),
            os.path.join(DEF_TTT_DIR,'gt_txt','train'),
            image_size=image_size,)
    elif(args.dataset=="msra"):
        train_dataset = MSRA(
            os.path.join(DEF_MSRA_DIR,'train'),
            os.path.join(DEF_MSRA_DIR,'train'),
            image_size=image_size)
    elif(args.dataset=="ic19"):
        train_dataset = ICDAR19(
            os.path.join(DEF_IC19_DIR,'images','train'),
            os.path.join(DEF_IC19_DIR,'gt_txt','train'),
            image_size=image_size)
    elif(args.dataset=="ic15"):
        train_dataset = ICDAR15(
            os.path.join(DEF_IC15_DIR,'images','train'),
            os.path.join(DEF_IC15_DIR,'gt_txt','train'),
            image_size=image_size)
    else:
        train_dataset = ICDAR13(
            os.path.join(DEF_IC13_DIR,'images','train'),
            os.path.join(DEF_IC13_DIR,'gt_txt','train'),
            os.path.join(DEF_IC13_DIR,'gt_pixel','train'),
            image_size=image_size,)


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
    time_c = datetime.now()
    total_step = len(train_loader)
    kernel = np.ones((3,3),dtype=np.uint8)
    for epoch in range(args.epoch):
        for i, sample in enumerate(train_loader):
            x = sample['image']
            b_have_mask = bool('mask' in sample)
            if(b_have_mask):
                mask = sample['mask']
            boxes = sample['box']
            if(random_b):
                angle = (np.random.random()-0.5)*2*5
                angle += 90*int(np.random.random()*4)
                h, w = x.shape[-3:-1]
                Mr = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1.0)
                Mr = np.concatenate((Mr,np.array([[0,0,1]],dtype=Mr.dtype)),0)

                scale_x = (np.random.random()*0.2+0.9)
                scale_y = (np.random.random()*0.2+0.9)
                Ms = np.array([[scale_x,0,0],[0,scale_y,0],[0,0,1]],dtype=Mr.dtype)

                shift_x = (np.random.random()-0.5)*2*7
                shift_y = (np.random.random()-0.5)*2*7
                Mt = np.array([[1,0,shift_x],[0,1,shift_y],[0,0,1]],dtype=Mr.dtype)
                
                Mtsr = np.dot(Mt,np.dot(Ms,Mr))
                image = x.numpy()
                image = np.stack([cv2.warpAffine(bt_image, Mtsr[:-1], (w, h)) for bt_image in image],0).astype(np.uint8)
                x = torch.from_numpy(image)

                boxes = [np_apply_matrix_to_pts(Mtsr,bth_box) if(bth_box.shape[0]>0)else bth_box for bth_box in boxes]
                if(b_have_mask):
                    mask_np = mask.numpy()
                    mask_np = np.stack([cv2.warpAffine(bt_mask, Mtsr[:-1], (w, h)) for bt_mask in mask],0).astype(np.uint8)
                    if(len(mask_np.shape)==3):
                        mask_np = np.expand_dims(mask_np,-1)
                    mask = torch.from_numpy(mask_np)
                    if(args.bxrefine):
                        refine_boxes = []
                        for batchi in range(x.shape[0]):
                            batchbox = boxes[batchi]
                            batchmask = mask_np[batchi,:,:,0]
                            refine_boxes.append(cv_refine_box_by_binary_map(batchbox,batchmask))
                        boxes = refine_boxes
                # boxes = np.array(tmp)

            if(b_have_mask):
                if(mask.shape[-1]==3): mask = mask[:,:,:,0:1]

                msak_np = mask.numpy()[:,:,:,0]
                msak_np = np.where(msak_np>0,255,0).astype(np.uint8)
                edge_np = [cv2.dilate(cv2.Canny(bth,100,200),kernel) for bth in msak_np]
                # np (batch,h,w,1)-> torch (batch,1,h,w)
                edge_np = np.stack(edge_np,0)
                
                edge = torch.from_numpy(np.expand_dims(edge_np,-1)).cuda(non_blocking=True)
                edge = (edge>0).float().permute(0,3,1,2)

                mask = mask.cuda(non_blocking=True)
                mask = (mask>0).float().permute(0,3,1,2)

            xnor = x.float().cuda(non_blocking=True)
            xnor = torch_img_normalize(xnor).permute(0,3,1,2)
            
            # Forward pass
            pred,feat = model(xnor)
            loss_dict = {}
            if('mask' in sample and DEF_BOOL_TRAIN_MASK):
                loss_dict['mask_loss'] = criterion(pred[:,DEF_START_MASK_CH+0], mask)
                loss_dict['edge_loss'] = criterion(pred[:,DEF_START_MASK_CH+1], edge)

            region_mask_np = []
            region_mask_bin_np = []
            boundary_mask_bin_np = []
            for batch_boxes in boxes:
                # (h,w)
                if(batch_boxes.shape[0]==0):
                    gas_map = np.zeros(x.shape[-3:-1],dtype=np.float32)
                    blmap = np.zeros(pred.shape[-2:],dtype=np.uint8)
                    boundadrymap = np.zeros(pred.shape[-2:],dtype=np.uint8)
                else:
                    gas_map,blmap = cv_gen_gaussian_by_poly(batch_boxes,x.shape[-3:-1],return_mask=True)
                    blmap = np.where(blmap>0,255,0).astype(np.uint8)
                    blmap = cv2.resize(blmap,(pred.shape[-1],pred.shape[-2]))
                    boundadrymap = blmap-cv2.erode(blmap,kernel,iterations=2)
        
                region_mask_np.append(gas_map)
                region_mask_bin_np.append(blmap)
                boundary_mask_bin_np.append(boundadrymap)

            if(DEF_BOOL_TRAIN_MASK):
                region_mask_np = np.expand_dims(np.stack(region_mask_np,0),-1).astype(np.float32)
                region_mask = torch.from_numpy(region_mask_np).cuda(non_blocking=True).permute(0,3,1,2)
                loss_dict['region_loss'] = criterion(pred[:,DEF_START_MASK_CH+2], region_mask)
            if(DEF_BOOL_TRAIN_CE):
                region_mask_bin_np = np.stack(region_mask_bin_np,0).astype(np.uint8)
                boundary_mask_bin_np = np.stack(boundary_mask_bin_np,0).astype(np.uint8)

                ce_y = np.zeros(region_mask_bin_np.shape,dtype=np.int64)
                ce_y[region_mask_bin_np>0]=1
                ce_y[boundary_mask_bin_np>0]=2
                ce_y_torch = torch.from_numpy(ce_y).cuda(non_blocking=True)
                loss_dict['cls_loss'] = criterion_ce(pred[:,DEF_START_CE_CH:], ce_y_torch)
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

            if(logger and rank==0):
                logger.add_scalar('Loss/total loss', loss.item(), epoch*total_step+i)
                for keyn,value in loss_dict.items():
                    logger.add_scalar('Loss/{}'.format(keyn), value.item(), epoch*total_step+i)

                logger.flush()


        if(rank == 0):
            time_usage = datetime.now() - time_c
            time_c = datetime.now()
            log.write('Epoch [{}/{}], Loss: {:.4f}\n'.format(epoch + 1, args.epoch,loss.item()))
            try:                         
                log.write("Time usage: {} Day {} Second.\n\n".format(time_usage.days,time_usage.seconds))
            except:
                log.write("Time usage: {} Day {} Second.\n\n".format(time_usage.day,time_usage.second))
            log.flush()

        if(args.save and rank == 0):
            # model_cpu = model.cpu()
            print("Saving model at {}...".format(args.save))
            torch.save(model.state_dict(),args.save)
        if(logger and rank==0):
            log_i,box_num = 0,boxes[0].shape[0]
            for i,o in enumerate(boxes):
                if(box_num<o.shape[0]):
                    log_i=i
                    box_num = o.shape[0]
            if(DEF_BOOL_TRAIN_MASK):
                pred_mask = pred[log_i,DEF_START_MASK_CH+0].to('cpu').detach().numpy()
                pred_edge = pred[log_i,DEF_START_MASK_CH+1].to('cpu').detach().numpy()
                pred_regi = pred[log_i,DEF_START_MASK_CH+2].to('cpu').detach().numpy()
                pred_mask = (pred_mask*255.0).astype(np.uint8)
                pred_edge = (pred_edge*255.0).astype(np.uint8)
                pred_regi = cv_heatmap(pred_regi)
                pred_mask = np.stack([pred_mask,pred_mask,pred_mask],-1)
                pred_edge = np.stack([pred_edge,pred_edge,pred_edge],-1)
                smx = cv2.resize(x[log_i].numpy().astype(np.uint8),(pred_edge.shape[1],pred_edge.shape[0]))
                smx = cv_mask_image(smx,pred_regi)
                line = np.ones((smx.shape[0],3,3),dtype=np.uint8)*255
                img = np.concatenate((smx,line,pred_mask,line,pred_edge),-2)
                logger.add_image('Prediction', img, epoch,dataformats='HWC')
                if(b_have_mask):
                    gt_mask = np.stack([msak_np[log_i],msak_np[log_i],msak_np[log_i]],-1)
                    gt_edge = np.stack([edge_np[log_i],edge_np[log_i],edge_np[log_i]],-1)
                    gt_regi = cv_heatmap(region_mask_np[log_i,:,:,0])
                    smx = cv2.resize(x[log_i].numpy().astype(np.uint8),(gt_mask.shape[1],gt_mask.shape[0]))
                    smx = cv_mask_image(smx,gt_regi)
                    line = np.ones((smx.shape[0],3,3),dtype=np.uint8)*255
                    img = np.concatenate((smx,line,gt_mask,line,gt_edge),-2)
                    logger.add_image('GT', img, epoch,dataformats='HWC')

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
                labels = torch.argmax(pred[log_i:log_i+1,DEF_START_CE_CH:],dim=1)
                labels = labels[0].cpu().detach().numpy()
                gtlabels = ce_y[log_i]
                labels = cv_labelmap(labels)
                gtlabels = cv_labelmap(gtlabels)
                if(gtlabels.shape[:-1]!=labels.shape[:-1]):
                    gtlabels = cv2.resize(gtlabels,(labels.shape[1],labels.shape[0]))
                line = np.ones((labels.shape[0],3,3),dtype=np.uint8)*255
                img = cv2.resize(x[log_i].numpy().astype(np.uint8),(gtlabels.shape[1],gtlabels.shape[0]))
                img = np.concatenate((img,line,gtlabels,line,labels),-2)
                logger.add_image('Labels Image|GT|Pred', img, epoch,dataformats='HWC')
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
    parser.add_argument('--opt', help='PKL path or name of optimizer.',default=tcfg['OPT'])
    parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
    parser.add_argument('--net', help='Choose noework.', default='vgg_mask')
    parser.add_argument('--tracker', type=str,help='Choose tracker.')
    parser.add_argument('--save', type=str, help='Set --save file_dir if want to save network.')
    parser.add_argument('--load', type=str, help='Set --load file_dir if want to load network.')
    parser.add_argument('--name', help='Name of task.')
    parser.add_argument('--dataset', help='Choose dataset: ctw/svt/ttt.', default=tcfg['DATASET'])
    parser.add_argument('--batch', type=int, help='Batch size.',default=tcfg['BATCH'])
    parser.add_argument('--learnrate', type=float, help='Learning rate.',default=tcfg['LR'])
    parser.add_argument('--epoch', type=int, help='Epoch size.',default=tcfg['EPOCH'])
    parser.add_argument('--random', type=int, help='Set 1 to enable random change.',default=0)
    parser.add_argument('--bxrefine', help='Set --debug if want to debug.', action="store_true")

    args = parser.parse_args()
    args.net = args.net.lower()
    args.dataset = args.dataset.lower() if(args.dataset)else args.dataset
    args.tracker = args.tracker.lower() if(args.tracker)else args.tracker
    
    # args.debug = True
    # args.random=True
    # args.batch=2
    summarize = "Start when {}.\n".format(datetime.now().strftime("%Y%m%d-%H%M%S")) +\
        "Working DIR: {}\n".format(DEF_WORK_DIR)+\
        "Running with: \n"+\
        "\t Epoch size: {},\n\t Batch size: {}.\n".format(args.epoch,args.batch)+\
        "\t Network: {}.\n".format(args.net)+\
        "\t Optimizer: {}.\n".format(args.opt)+\
        "\t Dataset: {}.\n".format(args.dataset)+\
        "\t Init learning rate: {}.\n".format(args.learnrate)+\
        "\t Taks name: {}.\n".format(args.name if(args.name)else 'None')+\
        "\t Load network: {}.\n".format(args.load if(args.load)else 'No')+\
        "\t Save network: {}.\n".format(args.save if(args.save)else 'No')+\
        "========\n"
        # "\t Train mask: {}.\n".format('Yes' if(DEF_BOOL_TRAIN_MASK)else 'No')+\
        # "\t Train classifier: {}.\n".format('Yes' if(DEF_BOOL_TRAIN_CE)else 'No')+\
    print(summarize)
    if(args.multi_gpu):
        gpus = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(init_process, nprocs=gpus, args=(gpus,train,args,))
    else:
        train(0, 1, args)
    # main()