import os
import sys
import platform
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
# =================Torch=======================
import torch
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# =================Local=======================
from lib.dataloader.base import split_dataset_cls_to_train_eval
from lib.dataloader.total import Total
from lib.dataloader.icdar import *
from lib.dataloader.msra import MSRA
from lib.model.pixel_map import PIXLSTM
from lib.loss.mseloss import *
from lib.utils.img_hlp import *
from lib.utils.log_hlp import *
from dirs import *

DEF_MOD_RATE = 0.3
DEF_WAVE_FUNC = lambda x: np.cos(2*x*np.pi)*DEF_MOD_RATE+1-DEF_MOD_RATE
DEF_LSTM_STATE_SIZE=(322,322)

def train(args):
    image_size=(640, 640)
    time_start = datetime.now()
    time_cur = time_start
    work_dir = os.path.join(DEF_WORK_DIR,'log')
    if(args.name):
        work_dir = os.path.join(work_dir,args.name)
    logger = SummaryWriter(os.path.join(work_dir,time_start.strftime("%Y%m%d-%H%M%S"))) if(not args.debug)else None
    log = sys.stdout

    model = PIXLSTM(mask_ch=2,basenet=args.basenet,min_upc_ch=128,min_map_ch=32,
        include_final=False,pretrained=True).float()

    if(args.load and os.path.exists(args.load)):
        log.write("Load parameters from {}.\n".format(args.load))
        model.load_state_dict(copyStateDict(torch.load(args.load)))
    model = model.cuda()
    
    criterion_mask = MSE_2d_Loss(pixel_sum=False).cuda()
    if(os.path.exists(args.opt)):
        optimizer = torch.load(args.opt)
    elif(args.opt.lower()=='adam'):
        optimizer = optim.Adam(model.parameters(), lr=args.learnrate, weight_decay=5e-4)
    elif(args.opt.lower() in ['adag','adagrad']):
        optimizer = optim.Adagrad(model.parameters(), lr=args.learnrate, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learnrate, momentum=0.8, weight_decay=5e-4)
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
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=train_dataset.default_collate_fn,)
    for epoch in range(args.epoch):
        for stepi, sample in enumerate(train_loader):
            x = sample['image']
            image = x.numpy()
            step_sample_list = []
            for fromei in range(args.maxstep):
                
            frame_bx_dict = sample['gt']
            p_keys = list(frame_bx_dict.keys())
            p_keys.sort()
            bxid_dict = defaultdict(list)
            for fm in frame_bx_dict:
                for bx in frame_bx_dict[fm]:
                    bxid_dict[int(bx[0])].append((fm,bx[1:].reshape(-1,2)))
            bxid_appear_dict = {bxid:None for bxid in bxid_dict}

            model.init_state(shape=DEF_LSTM_STATE_SIZE,batch_size=1)
            fm_cnt,proc_cnt = 0,0
            try:
            # if(1):
                while(vdo.isOpened()):
                    ret, x = vdo.read()
                    if(ret==False):
                        break
                    if(fm_cnt<p_keys[0] or fm_cnt%args.skiprate):
                        # skip the initial non text frams
                        fm_cnt+=1
                        continue
                    
                    x = cv2.resize(x,image_size[::-1])
                    xnor = torch.from_numpy(np.expand_dims(x,0)).float().cuda()
                    xnor = torch_img_normalize(xnor).permute(0,3,1,2)

                    # Gen current mask
                    if(fm_cnt in frame_bx_dict):
                        bxids_boxes = frame_bx_dict[fm_cnt]
                        bxids = bxids_boxes[:,0].astype(np.uint16).tolist()
                        boxes = bxids_boxes[:,1:].reshape(bxids_boxes.shape[0],-1,2)
                        region_mask_np = cv_gen_gaussian_by_poly(boxes,x.shape[:-1])

                    else:
                        bxids = []
                        region_mask_np = np.zeros(x.shape[:-1],dtype=np.float32)

                    assert isinstance(region_mask_np,np.ndarray)

                    if(args.linear):
                        # Gen previous mask nad apply weight decay
                        rmlist = []
                        for bxid in bxid_appear_dict:
                            if(fm_cnt>=bxid_dict[bxid][-1][0]):
                                rmlist.append(bxid)
                            else:
                                sp_box,ep_box = bxid_dict[bxid][0][1],bxid_dict[bxid][-1][1]
                                sp_fm,ep_fm = bxid_dict[bxid][0][0],bxid_dict[bxid][-1][0]
                                for i in range(len(bxid_dict[bxid])-1):
                                    if(bxid_dict[bxid][i][0]<=fm_cnt and bxid_dict[bxid][i+1][0]>=fm_cnt):
                                        sp_box,ep_box = bxid_dict[bxid][i][1],bxid_dict[bxid][i+1][1]
                                        sp_fm,ep_fm = bxid_dict[bxid][i][0],bxid_dict[bxid][i+1][0]
                                        break
                                mov_fact = (fm_cnt-sp_fm)/(ep_fm-sp_fm)
                                cur_box = (ep_box-sp_box)*mov_fact + sp_box
                                cur_box = cur_box.reshape(-1,2)
                                mov_mask = cv_gen_gaussian_by_poly(cur_box,x.shape[:-1])
                                mov_mask *= DEF_WAVE_FUNC(mov_fact)

                            region_mask_np+=mov_mask

                    pred,feat = model(xnor)
                    loss_dict = {}
                    region_mask = torch.from_numpy(region_mask_np.reshape(1,1,region_mask_np.shape[0],region_mask_np.shape[1])).float().cuda()
                    loss_dict['region_loss'] = criterion_mask(pred[:,0:1],region_mask)

                    loss = 0.0
                    for keyn,value in loss_dict.items():
                        loss+=value
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if(DEF_FLUSH_COUNT>0 and (proc_cnt+1)%DEF_FLUSH_COUNT==0):
                        model.init_state(shape=DEF_LSTM_STATE_SIZE,batch_size=1)

                    if(logger):
                        region_np = pred[0,0].detach().cpu().numpy()
                        logger.add_image('Region: s{},e{}'.format(stepi,epoch),concatenate_images([x,cv_heatmap(region_np)]),fm_cnt,dataformats='HWC')
                    fm_cnt += 1
                    proc_cnt += 1
            except Exception as e:
                sys.stdout.write("Err at {}, frame {}, frame processed {}.\nErr: {}\n".format(sample['name'],fm_cnt,proc_cnt+1,str(e)))
                sys.stdout.flush()
        time_usage = datetime.now() - time_cur
        time_cur = datetime.now()
        print_epoch_log(epoch + 1, args.epoch,loss.item(),time_usage)

        if(args.save):
            log.write("Saving model at {}...\n".format(args.save))
            if(not os.path.exists(os.path.dirname(args.save))):
                os.makedirs(os.path.dirname(args.save))
            torch.save(model.state_dict(),args.save)
        log.flush()
    
    # finish
    time_usage = datetime.now() - time_start
    try:
        log.write("Total time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    except:
        log.write("Total time usage: {} Day {} Second.\n".format(time_usage.day,time_usage.second))
    log.close()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_gpu', help='Set --multi_gpu to enable multi gpu training.',action="store_true")
    parser.add_argument('--opt', help='PKL path or name of optimizer.',default='adag')
    parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
    parser.add_argument('--eval', help='Set --eval to enable eval.', action="store_true")
    parser.add_argument('--basenet', help='Choose base noework.', default='mobile')
    parser.add_argument('--tracker', type=str,help='Choose tracker.')
    parser.add_argument('--save', type=str, help='Set --save file_dir if want to save network.')
    parser.add_argument('--load', type=str, help='Set --load file_dir if want to load network.')
    parser.add_argument('--name', help='Name of task.')
    parser.add_argument('--dataset', help='Choose dataset: Minetto/icv15.', default='Minetto')
    parser.add_argument('--learnrate', type=str, help='Learning rate.',default="0.001")
    parser.add_argument('--epoch', type=str, help='Epoch size.',default="10")
    parser.add_argument('--lr_decay', help='Set --lr_decay to enbable learning rate decay.', action="store_true")
    # LSTM specific
    parser.add_argument('--maxstep', type=int, help='Max lenth of single image.',default=5)

    args = parser.parse_args()
    args.dataset = args.dataset.lower() if(args.dataset)else args.dataset
    args.tracker = args.tracker.lower() if(args.tracker)else args.tracker
    args.maxstep = max(args.maxstep,3)
    # args.debug = True

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
            "\t Epoch size: {}.\n".format(args.epoch)+\
            "\t Base network: {}.\n".format(args.basenet)+\
            "\t Optimizer: {}.\n".format(cur_opt)+\
            "\t LR decay: {}.\n".format('Yes' if(args.lr_decay)else 'No')+\
            "\t Dataset: {}.\n".format(cur_dataset)+\
            "\t Move step per-image: {}.\n".format(args.maxstep)+\
            "\t Init learning rate: {}.\n".format(cur_learnrate)+\
            "\t Taks name: {}.\n".format(args.name if(args.name)else 'None')+\
            "\t Load network: {}.\n".format(args.load if(args.load)else 'No')+\
            "\t Save network: {}.\n".format(args.save if(args.save)else 'No')+\
            "\t Linear: {}.\n".format('Yes' if(args.linear)else 'No')+\
            "========\n"
        print(summarize)

        args.opt = cur_opt
        args.dataset = cur_dataset
        args.epoch = cur_epoch
        args.learnrate = cur_learnrate

        train(args)
        last_save = args.save