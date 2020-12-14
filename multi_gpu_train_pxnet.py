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
from lib.model.pixel_map import PIX_TXT,VGG_PXMASK,VGGUnet_PXMASK
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
    image_size = (640,640)
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
    model = VGGUnet_PXMASK(pretrained=True,padding=False,include_b0=True).float()
    if(args.load and os.path.exists(args.load)):
        model.load_state_dict(copyStateDict(torch.load(args.load)))
    model = model.cuda(rank)

    # define loss function (criterion) and optimizer
    criterion = MSE_2d_Loss(pixel_sum=False).cuda(rank)

    if(os.path.exists(args.opt)):
        optimizer = torch.load(args.opt)
    elif(args.opt.lower()=='adam'):
        optimizer = optim.Adam(model.parameters(), lr=args.learnrate, weight_decay=tcfg['OPT_DEC'])
    elif(args.opt.lower() in ['adag','adagrad']):
        optimizer = optim.Adagrad(model.parameters(), lr=args.learnrate, weight_decay=tcfg['OPT_DEC'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learnrate, momentum=tcfg['MMT'], weight_decay=tcfg['OPT_DEC'])
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # Data loading code

    if(args.dataset=="ttt"):
        train_dataset = Total(
            os.path.join(DEF_TTT_DIR,'images','train'),
            os.path.join(DEF_TTT_DIR,'gt_pixel','train'),
            os.path.join(DEF_TTT_DIR,'gt_txt','train'),
            image_size=image_size,)
        x_input_function = train_dataset.x_input_function
        y_input_function = None
    else:
        train_dataset = ICDAR13(
            os.path.join(DEF_IC13_DIR,'images','train'),
            os.path.join(DEF_IC13_DIR,'gt_txt','train'),
            os.path.join(DEF_IC13_DIR,'gt_pixel','train'),
            image_size=image_size,)
        x_input_function = train_dataset.x_input_function
        y_input_function = None


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
            mask = sample['mask']

            if(random_b):
                angle = (np.random.random()-0.5)*2*5
                h, w = x.shape[-3:-1]
                Mr = cv2.getRotationMatrix2D((w-1/2, h-1/2), angle, 1.0)
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
                mask = mask.numpy()
                mask = np.stack([cv2.warpAffine(bt_mask, Mtsr[:-1], (w, h)) for bt_mask in mask],0).astype(np.uint8)
                if(len(mask.shape)==3):
                    mask = np.expand_dims(mask,-1)
                mask = torch.from_numpy(mask)

            if(mask.shape[-1]==3):
                mask = mask[:,:,:,0:1]
            msak_np = mask.numpy()[:,:,:,0]
            msak_np = np.where(msak_np>0,255,0).astype(np.uint8)
            edge_np = [cv2.dilate(cv2.Canny(bth,100,200),kernel) for bth in msak_np]
            # np (batch,h,w,1)-> torch (batch,1,h,w)
            edge_np = np.expand_dims(np.stack(edge_np,0),-1)
            edge = torch.from_numpy(edge_np).cuda(non_blocking=True)
            edge = (edge>0).float().permute(0,3,1,2)

            mask = mask.cuda(non_blocking=True)
            mask = (mask>0).float().permute(0,3,1,2)

            xnor = x.float().cuda(non_blocking=True)
            xnor = torch_img_normalize(xnor).permute(0,3,1,2)

            # Forward pass
            pred,feat = model(xnor)
            mask_loss = criterion(pred[:,0], mask)
            edge_loss = criterion(pred[:,1], edge)
            region_mask_np = []
            for batch_boxes in sample['box']:
                region = [cv_gen_gaussian_by_poly(box,x.shape[-3:-1]) for box in batch_boxes]
                region_mask_np.append(sum(region))
            if(random_b):
                region_mask_np = np.stack([cv2.warpAffine(rg_mask, Mtsr[:-1], (w, h)) for rg_mask in region_mask_np],0)
                region_mask_np = np.expand_dims(region_mask_np,-1).astype(np.float32)
            else:
                region_mask_np = np.expand_dims(np.stack(region_mask_np,0),-1).astype(np.float32)
            region_mask = torch.from_numpy(region_mask_np).cuda(non_blocking=True).permute(0,3,1,2)
            region_loss = criterion(pred[:,2], region_mask)
            
            loss = edge_loss+mask_loss+region_loss
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(logger and rank==0):
                logger.add_scalar('Loss/total loss', loss.item(), epoch*total_step+i)
                logger.add_scalar('Loss/mask_loss', mask_loss.item(), epoch*total_step+i)
                logger.add_scalar('Loss/edge_loss', edge_loss.item(), epoch*total_step+i)
                logger.add_scalar('Loss/region_loss', region_loss.item(), epoch*total_step+i)
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
            print("Saving model at {}...".format(args.save+'.pth'))
            torch.save(model.state_dict(),args.save+'.pth')
        if(logger and rank==0):
            pred_mask = pred[0,0].to('cpu').detach().numpy()
            pred_edge = pred[0,1].to('cpu').detach().numpy()
            pred_regi = pred[0,2].to('cpu').detach().numpy()
            pred_mask = (pred_mask*255.0).astype(np.uint8)
            pred_edge = (pred_edge*255.0).astype(np.uint8)
            pred_regi = cv_heatmap(pred_regi)
            pred_mask = np.stack([pred_mask,pred_mask,pred_mask],-1)
            pred_edge = np.stack([pred_edge,pred_edge,pred_edge],-1)
            smx = cv2.resize(x[0].numpy().astype(np.uint8),(pred_edge.shape[1],pred_edge.shape[0]))
            smx = cv_mask_image(smx,pred_regi)
            line = np.ones((smx.shape[0],3,3),dtype=smx.dtype)*255
            img = np.concatenate((smx,line,pred_mask,line,pred_edge),-2)
            logger.add_image('Prediction', img, epoch,dataformats='HWC')

            gt_mask = np.stack([msak_np[0],msak_np[0],msak_np[0]],-1)
            gt_edge = np.stack([edge_np[0,:,:,0],edge_np[0,:,:,0],edge_np[0,:,:,0]],-1)
            gt_regi = cv_heatmap(region_mask_np[0,:,:,0])
            smx = cv2.resize(x[0].numpy().astype(np.uint8),(gt_mask.shape[1],gt_mask.shape[0]))
            smx = cv_mask_image(smx,gt_regi)
            line = np.ones((smx.shape[0],3,3),dtype=smx.dtype)*255
            img = np.concatenate((smx,line,gt_mask,line,gt_edge),-2)
            logger.add_image('GT', img, epoch,dataformats='HWC')

            # b0_mask = feat.b0_mask[0].to('cpu').permute(1,2,0).detach().numpy()
            # b1_mask = feat.b1_mask[0].to('cpu').permute(1,2,0).detach().numpy()
            # b2_mask = feat.b2_mask[0].to('cpu').permute(1,2,0).detach().numpy()
            # b3_mask = feat.b3_mask[0].to('cpu').permute(1,2,0).detach().numpy()
            # b4_mask = feat.b4_mask[0].to('cpu').permute(1,2,0).detach().numpy()

            # b1_mask = cv2.resize(b1_mask,(b0_mask.shape[1],b0_mask.shape[0]))
            # b2_mask = cv2.resize(b2_mask,(b0_mask.shape[1],b0_mask.shape[0]))
            # b3_mask = cv2.resize(b3_mask,(b0_mask.shape[1],b0_mask.shape[0]))
            # b4_mask = cv2.resize(b4_mask,(b0_mask.shape[1],b0_mask.shape[0]))
            # line = np.ones((b0_mask.shape[0],3,3),dtype=np.uint8)*255
            # img = np.concatenate((
            #     cv_heatmap(b0_mask[:,:,0]),line,
            #     cv_heatmap(b1_mask[:,:,0]),line,
            #     cv_heatmap(b2_mask[:,:,0]),line,
            #     cv_heatmap(b3_mask[:,:,0]),line,
            #     cv_heatmap(b4_mask[:,:,0])),-2)
            # logger.add_image('B0|B1|B2|B3|B4 mask', img, epoch,dataformats='HWC')
            # img = np.concatenate((
            #     cv_heatmap(b0_mask[:,:,1]),line,
            #     cv_heatmap(b1_mask[:,:,1]),line,
            #     cv_heatmap(b2_mask[:,:,1]),line,
            #     cv_heatmap(b3_mask[:,:,1]),line,
            #     cv_heatmap(b4_mask[:,:,1])),-2)
            # logger.add_image('B0|B1|B2|B3|B4 edge', img, epoch,dataformats='HWC')
            # img = np.concatenate((
            #     cv_heatmap(b0_mask[:,:,2]),line,
            #     cv_heatmap(b1_mask[:,:,2]),line,
            #     cv_heatmap(b2_mask[:,:,2]),line,
            #     cv_heatmap(b3_mask[:,:,2]),line,
            #     cv_heatmap(b4_mask[:,:,2])),-2)
            # logger.add_image('B0|B1|B2|B3|B4 region', img, epoch,dataformats='HWC')
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
    parser.add_argument('--opt', help='PKL path or name of optimizer.',default=tcfg['OPT'])
    parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
    parser.add_argument('--save', type=str, help='Set --save file_dir if want to save network.')
    parser.add_argument('--load', type=str, help='Set --load file_dir if want to load network.')
    parser.add_argument('--name', help='Name of task.')
    parser.add_argument('--dataset', help='Choose dataset: ctw/svt/ttt.', default=tcfg['DATASET'])
    parser.add_argument('--batch', type=int, help='Batch size.',default=tcfg['BATCH'])
    parser.add_argument('--learnrate', type=float, help='Learning rate.',default=tcfg['LR'])
    parser.add_argument('--epoch', type=int, help='Epoch size.',default=tcfg['EPOCH'])
    parser.add_argument('--random', type=int, help='Set 1 to enable random change.',default=0)

    args = parser.parse_args()
    summarize = "Start when {}.\n".format(datetime.now().strftime("%Y%m%d-%H%M%S")) +\
        "Working DIR: {}\n".format(DEF_WORK_DIR)+\
        "Running with: \n"+\
        "\t Epoch size: {},\n\t Batch size: {}.\n".format(args.epoch,args.batch)+\
        "\t Network: {}.\n".format('PixelTXT')+\
        "\t Optimizer: {}.\n".format(args.opt)+\
        "\t Dataset: {}.\n".format(args.dataset)+\
        "\t Init learning rate: {}.\n".format(args.learnrate)+\
        "\t Taks name: {}.\n".format(args.name if(args.name)else 'None')+\
        "\t Load network: {}.\n".format(args.load if(args.load)else 'No')+\
        "\t Save network: {}.\n".format(args.save if(args.save)else 'No')+\
        "========\n"
    print(summarize)
    gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(init_process, nprocs=gpus, args=(gpus,train,args,))
    # main()