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
from lib.fr_craft import CRAFTTrainer
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
    work_dir = DEF_WORK_DIR
    work_dir = os.path.join(work_dir,'log')
    if(args.name):
        work_dir = os.path.join(work_dir,args.name)
    if(rank==0):
        logger = SummaryWriter(work_dir)
    else:
        logger = None

    batch_size = args.batch
    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    dev = 'cuda:{}'.format(rank)
    model = PIX_TXT(pretrained=True).float()
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
            image_size=(1080, 1080),)
        x_input_function = train_dataset.x_input_function
        y_input_function = None
    else:
        train_dataset = ICDAR13(
            os.path.join(DEF_IC13_DIR,'images','train'),
            os.path.join(DEF_IC13_DIR,'gt_txt','train'),
            os.path.join(DEF_IC13_DIR,'gt_pixel','train'),
            image_size=(1080, 1080),)
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

    start = datetime.now()
    total_step = len(train_loader)
    kernel = np.ones((3,3),dtype=np.uint8)
    for epoch in range(args.epoch):
        for i, sample in enumerate(train_loader):
            x = sample['image']
            mask = sample['mask']
            boxes = sample['box']
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
            pred,_ = model(xnor)
            mask_loss = criterion(pred[:,0], mask)
            edge_loss = criterion(pred[:,1], edge)
            loss = edge_loss+mask_loss
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(logger and rank==0):
                logger.add_scalar('Loss/total loss', loss.item(), epoch*total_step+i)
                logger.add_scalar('Loss/mask_loss', mask_loss.item(), epoch*total_step+i)
                logger.add_scalar('Loss/edge_loss', edge_loss.item(), epoch*total_step+i)
                logger.flush()


            if (i + 1) % 100 == 0 and rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epoch, i + 1, total_step,
                                                                         loss.item()))

        if(args.save and rank == 0):
            # model_cpu = model.cpu()
            print("Saving model at {}...".format(args.save+'.pth'))
            torch.save(model.state_dict(),args.save+'.pth')
        if(logger and rank==0):
            pred_mask = pred[0,0].to('cpu').detach().numpy()
            pred_edge = pred[0,1].to('cpu').detach().numpy()
            pred_mask = (pred_mask*255.0).astype(np.uint8)
            pred_edge = (pred_edge*255.0).astype(np.uint8)
            pred_mask = np.stack([pred_mask,pred_mask,pred_mask],-1)
            pred_edge = np.stack([pred_edge,pred_edge,pred_edge],-1)
            smx = cv2.resize(x[0].numpy().astype(np.uint8),(pred_edge.shape[1],pred_edge.shape[0]))
            line = np.ones((smx.shape[0],3,3),dtype=smx.dtype)*255
            img = np.concatenate((smx,line,pred_mask,line,pred_edge),-2)
            logger.add_image('Prediction', img, epoch,dataformats='HWC')

            gt_mask = np.stack([msak_np[0],msak_np[0],msak_np[0]],-1)
            gt_edge = np.stack([edge_np[0,:,:,0],edge_np[0,:,:,0],edge_np[0,:,:,0]],-1)
            smx = cv2.resize(x[0].numpy().astype(np.uint8),(gt_mask.shape[1],gt_mask.shape[0]))
            line = np.ones((smx.shape[0],3,3),dtype=smx.dtype)*255
            img = np.concatenate((smx,line,gt_mask,line,gt_edge),-2)
            logger.add_image('GT', img, epoch,dataformats='HWC')
            logger.flush()                                                       

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
    parser.add_argument('--net', help='Choose noework (craft).', default=tcfg['NET'])
    parser.add_argument('--name', help='Name of task.')
    parser.add_argument('--dataset', help='Choose dataset: ctw/svt/ttt.', default=tcfg['DATASET'])
    parser.add_argument('--datax', type=int, help='Dataset output width.',default=tcfg['IMG_SIZE'][0])
    parser.add_argument('--datay', type=int, help='Dataset output height.',default=tcfg['IMG_SIZE'][1])
    parser.add_argument('--step', type=int, help='Step size.',default=tcfg['STEP'])
    parser.add_argument('--batch', type=int, help='Batch size.',default=tcfg['BATCH'])
    parser.add_argument('--logstp', type=int, help='Log step size.',default=tcfg['LOGSTP'])
    parser.add_argument('--gpu', type=int, help='Set --gpu -1 to disable gpu.',default=0)
    parser.add_argument('--savestep', type=int, help='Save step size.',default=tcfg['SAVESTP'])
    parser.add_argument('--learnrate', type=float, help='Learning rate.',default=tcfg['LR'])
    parser.add_argument('--teacher', type=str, help='Set --teacher to pkl file.')
    parser.add_argument('--epoch', type=int, help='Epoch size.',default=tcfg['EPOCH'])

    args = parser.parse_args()
    gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(init_process, nprocs=gpus, args=(gpus,train,args,))
    # main()