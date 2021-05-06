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
from lib.model.pixel_map import PIXLSTM,PIXLSTM_Residual,PIXCNN
from lib.loss.mseloss import *
from lib.utils.img_hlp import *
from lib.utils.log_hlp import *
from dirs import *

DEF_MOD_RATE = 0.3
DEF_WAVE_FUNC = lambda x: np.cos(2*x*np.pi)*DEF_MOD_RATE+1-DEF_MOD_RATE
DEF_LSTM_STATE_SIZE=(322,322)

max_fscore = 0.0

def train(args):
    global max_fscore
    image_size=(640, 640)
    time_start = datetime.now()
    time_cur = time_start
    work_dir = os.path.join(DEF_WORK_DIR,'log')
    if(args.name):
        work_dir = os.path.join(work_dir,args.name)
    logger = SummaryWriter(os.path.join(work_dir,time_start.strftime("%Y%m%d-%H%M%S"))) if(not args.debug)else None
    log = sys.stdout

    model = PIXCNN(mask_ch=2,basenet=args.basenet,min_upc_ch=128,min_map_ch=32,
        include_final=False,pretrained=True).float()
    dshape = (1,model.final_predict_ch,DEF_LSTM_STATE_SIZE[0],DEF_LSTM_STATE_SIZE[1])

    try:
        model.lstm.Wci = nn.Parameter(torch.rand(dshape,dtype=torch.float32),requires_grad=True)
        model.lstm.Wcf = nn.Parameter(torch.rand(dshape,dtype=torch.float32),requires_grad=True)
        model.lstm.Wco = nn.Parameter(torch.rand(dshape,dtype=torch.float32),requires_grad=True)
    except:
        None

    if(args.load and os.path.exists(args.load)):
        log.write("Load parameters from {}.\n".format(args.load))
        model.load_state_dict(copyStateDict(torch.load(args.load)))
        model = model.cuda()

    model = model.cuda()
    _,d = next(iter(model.state_dict().items()))
    model_device,model_dtype = d.device,d.dtype

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
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=train_dataset.default_collate_fn,)
    if(args.eval):
        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                batch_size=args.batch,
                                                shuffle=True,
                                                num_workers=0,
                                                pin_memory=True,
                                                collate_fn=eval_dataset.default_collate_fn,)
    total_step = len(train_loader)
    distributes = defaultdict(list)
    
    for epoch in range(args.epoch):
        for stepi, sample in enumerate(train_loader):
            x = sample['image']
            bth_image = x.numpy()
            bth_boxes = sample['box']
            bth_sample = []
            seed = np.random
            for bthi in range(bth_image.shape[0]):
                image,box = bth_image[bthi],bth_boxes[bthi]
                # rotation in [0,360] * random in [-1,+1]
                rotate = min(360,args.maxstep * args.rotatev) *(seed.random()-0.5)*2
                # shift in [0,200] * random in [-1,+1]
                shiftx = min(200,args.maxstep * args.shiftv) * (seed.random()-0.5)*2
                shifty = min(200,args.maxstep * args.shiftv) * (seed.random()-0.5)*2
                # scale in [0.5,3] * random in [-1,+1]
                scalex = min(3,max(0.5,args.maxstep * args.scalev)) *(seed.random()-0.5)*2
                scaley = min(3,max(0.5,args.maxstep * args.scalev)) *(seed.random()-0.5)*2
                if(not args.random):
                    image_list,poly_xy_list,Ms = cv_gen_trajectory(image,args.maxstep,box,
                        rotate=rotate,shift=(shifty,shiftx),scale=(scaley,scalex)
                        ,blur=True,blur_rate=0.5,blur_ksize=15,blur_intensity=0.2,
                        )
                else:
                    image_list,poly_xy_list,Ms = [],[],[]
                    for _ in range(args.maxstep):
                        imgt,boxt,M = cv_random_image_process(image,box,False,
                        rotate=rotate,random_90=False,
                        shift = max(shiftx,shifty),
                        scale_weight = max(scalex,scaley)*0.2,scale_base = max(scalex,scaley)*0.9,
                        )
                        image_list.append(imgt)
                        poly_xy_list.append(boxt)
                        Ms.append(M)

                distributes['rotate'].append(rotate)
                distributes['shiftx'].append(shiftx)
                distributes['shifty'].append(shifty)
                distributes['scalex'].append(scalex)
                distributes['scaley'].append(scaley)

                if('text' in sample and '#' in sample['text'][bthi]):
                    gt_list = []
                    weight_mask_list = []
                    txt = sample['text'][bthi]
                    bgid = [i for i,o in enumerate(txt) if(o=='#')]
                    for poly_xy in poly_xy_list:
                        bgbxs = np.array([o for i,o in enumerate(poly_xy) if(i in bgid)])
                        fgbxs = np.array([o for i,o in enumerate(poly_xy) if(i not in bgid)])
                        weight_mask_list.append(cv_gen_binary_mask_by_poly(bgbxs,image_size,default_value=1,default_fill=0))
                        gt_list.append(cv_gen_gaussian_by_poly(fgbxs,image_size))
                else:
                    gt_list = [cv_gen_gaussian_by_poly(o,image_size) for o in poly_xy_list]
                    weight_mask_list = [np.ones(image_size,dtype=np.float32)]*len(gt_list)
                bth_sample.append((image_list,poly_xy_list,gt_list,weight_mask_list))
            
            bth_state = [(
                # lstmh
                torch.zeros(dshape,dtype = model_dtype, device=model_device),
                # lstmc
                torch.zeros(dshape,dtype = model_dtype, device=model_device),
                ) for _ in range(len(bth_sample))]
            
            loss_dict = {}
            bth_avg_loss = 0.0
            bth_pred_region = []
            for (lstmh,lstmc),(image_list,poly_xy_list,gt_list,weight_mask_list) in zip(bth_state,bth_sample):
                x_sequence = torch.tensor(image_list,dtype = model_dtype, device=model_device)
                y = torch.tensor(gt_list,dtype = model_dtype, device=model_device)
                y_mask = torch.tensor(weight_mask_list,dtype = model_dtype, device=model_device)

                xnor = torch_img_normalize(x_sequence)
                xnor = xnor.permute(0,3,1,2)
                avg_loss = 0.0
                try:
                    model.lstmh,model.lstmc=lstmh,lstmc
                except:
                    None
                pred_region = []
                for framei in range(xnor.shape[0]):
                    pred,feat = model(xnor[framei:framei+1])
                    avg_loss+=criterion_mask(pred[:,0:1],y[framei:framei+1],y_mask[framei:framei+1])
                    pred_region.append(pred[0,0].to('cpu').detach().numpy())

                bth_pred_region.append(pred_region)
                avg_loss/=xnor.shape[0]
                bth_avg_loss+=avg_loss

            loss_dict['region_loss']=bth_avg_loss/len(bth_sample)

            loss = 0.0
            for keyn,value in loss_dict.items():
                loss+=value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(logger):
                logger.add_scalar('Loss/total loss', loss.item(), epoch*total_step+stepi)
                for keyn,value in loss_dict.items():
                    logger.add_scalar('Loss/{}'.format(keyn), value.item(), epoch*total_step+stepi)
                if(args.lr_decay):
                    logger.add_scalar('LR rate', optimizer.param_groups[0]['lr'], epoch*total_step+stepi)
                logger.flush()
            if(args.debug):
                break

        # End of epoch
        if(args.eval):
            recall_list,precision_list = [],[]
            eval_log_dict = {}
            eval_log_szie = 10
            eval_log_cnt = 0
            with torch.no_grad():
                dshape = (1,model.final_predict_ch,DEF_LSTM_STATE_SIZE[0],DEF_LSTM_STATE_SIZE[1])
                model.lstmc=torch.zeros(dshape,dtype = model_dtype, device=model_device)
                model.lstmh=torch.zeros(dshape,dtype = model_dtype, device=model_device)

                for stepi, sample in enumerate(eval_loader):

                    x = sample['image']
                    bth_image = x.numpy()
                    bth_boxes = sample['box']
                    x_nor = torch_img_normalize(x).float().to(model_device).permute(0,3,1,2)

                    seed = np.random
                    for bthi in range(bth_image.shape[0]):
                        if('text' in sample):
                            txt = sample['text'][bthi]
                            boxes = bth_boxes[bthi]
                            if(len(boxes.shape)==2):
                                boxes = np.expand_dims(boxes,0)
                            bgbxs = np.array([boxes[i] for i in range(len(txt)) if(txt[i]=='#')])
                            fgbxs = np.array([boxes[i] for i in range(len(txt)) if(txt[i]!='#')])
                        else:
                            fgbxs=bth_boxes[bthi]
                            bgbxs=None

                        image = bth_image[bthi]
                        rotate = args.maxstep * args.rotatev *(seed.random()-0.5)*2.2
                        shiftx = args.maxstep * args.shiftv *(seed.random()-0.5)*2.2
                        shifty = args.maxstep * args.shiftv *(seed.random()-0.5)*2.2
                        scalex = args.maxstep * args.scalev *(seed.random()-0.5)*2.2
                        scaley = args.maxstep * args.scalev *(seed.random()-0.5)*2.2
                        image_list,_,Ms = cv_gen_trajectory(image,args.maxstep,
                            rotate=rotate,shift=(shifty,shiftx),scale=(scaley,scalex))

                        model.lstmc.zero_()
                        model.lstmh.zero_()
                        image_list_torch = torch.tensor(image_list,dtype = model_dtype, device=model_device)
                        image_list_nor = torch_img_normalize(image_list_torch).permute(0,3,1,2)
                        pred_np = []
                        for imgid in range(image_list_nor.shape[0]):
                            pred,_ = model(image_list_nor[imgid:imgid+1])
                            pred_np.append(pred[0,0].cpu().detach().numpy())
                        
                        fgbxs_list = []
                        recall_list_sg,precision_list_sg = [],[]
                        for M,region_np in zip(Ms,pred_np):
                            if(fgbxs is not None and len(fgbxs)>0):
                                cur_fgbxs = np_apply_matrix_to_pts(M,fgbxs)
                            else:
                                cur_fgbxs = None
                            fgbxs_list.append(cur_fgbxs)

                            region_np[region_np<0.2]=0.0
                            det_boxes, label_mask, label_list = cv_get_box_from_mask(region_np,region_mean_split=True)
                            if(det_boxes.shape[0]>0):
                                det_boxes = np_box_resize(det_boxes,region_np.shape[-2:],x.shape[-3:-1],'polyxy')
                                ids,mask_precision,mask_recall = cv_box_match(det_boxes,fgbxs,bgbxs,ovth=0.5)
                            else:
                                mask_precision,mask_recall=0.0,0.0
                            recall_list_sg.append(mask_recall)
                            precision_list_sg.append(mask_precision)
                        
                        recall = np.mean(recall_list_sg)
                        precision = np.mean(precision_list_sg)
                        recall_list.append(recall)
                        precision_list.append(precision)
                        fscore = 2*recall*precision/(recall+precision) if(recall+precision>0)else 0.0

                        if(fscore<0.5 and eval_log_cnt<eval_log_szie):
                            t = []
                            for img,bx,region_np in zip(image_list,fgbxs_list,pred_np):
                                gt = cv_gen_gaussian_by_poly(bx,image_size)
                                t.append(concatenate_images([img,cv_heatmap(gt),cv_heatmap(region_np)]))
                            eval_log_dict['Eval/ Epoch {}, step {}.'.format(epoch+1,stepi*bth_image.shape[0]+bthi)]=t
                            eval_log_cnt+=1
                    if(args.debug):
                        break
            recall,precision=np.mean(recall_list),np.mean(precision_list)
            fscore = 2*recall*precision/(precision+recall) if((precision+recall)>0)else 0
            log.write("Recall|Precision|F-score in t0: {:3.2f}%|{:3.2f}%|{:3.2f}%.\n".format(recall*100,precision*100,fscore*100))
            log.flush()
            if(logger):
                logger.add_scalar('Eval/recall', recall, epoch)
                logger.add_scalar('Eval/precision', precision, epoch)
                logger.add_scalar('Eval/F-score', fscore, epoch)
                for key in eval_log_dict:
                    if(isinstance(eval_log_dict[key],list)):
                        for i, o in enumerate(eval_log_dict[key]):
                            logger.add_image(key,o,i,dataformats='HWC')
                    else:
                        logger.add_image(key,eval_log_dict[key],0,dataformats='HWC')
                logger.flush()
            if(fscore>max_fscore):
                max_fscore = fscore
                fmdir,fmname = os.path.split(args.save)
                fmname = 'max_eval_'+fmname
                finalname = os.path.join(fmdir,fmname)
                log.write("Saving model at {}...\n".format(finalname))
                if(not os.path.exists(os.path.dirname(fmdir))):
                    os.makedirs(os.path.dirname(args.save))
                torch.save(model.state_dict(),finalname)

        if(logger):
            for k,v in distributes.items():
                logger.add_histogram('Parameters Distribution/'+k,np.array(v),epoch)
            distributes = defaultdict(list)

            logger.flush()

        time_usage = datetime.now() - time_cur
        time_cur = datetime.now()
        print_epoch_log(epoch, args.epoch,loss.item(),time_usage)
        if(logger and (epoch+1)%5==0):
            image_list,pred_region_list,gt_list=bth_sample[0][0],bth_pred_region[0],bth_sample[0][2]
            for framei in range(len(bth_pred_region[0])):
                img = concatenate_images([image_list[framei],cv_heatmap(pred_region_list[framei]),cv_heatmap(gt_list[framei])])
                logger.add_image('Region: e{}'.format(epoch),img,framei,dataformats='HWC')
            logger.flush()

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

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', help='PKL path or name of optimizer.',default='adag')
    parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
    parser.add_argument('--eval', help='Set --eval to enable eval.', action="store_true")
    parser.add_argument('--batch', type=int, help='Batch size.',default=4)
    parser.add_argument('--basenet', help='Choose base noework.', default='mobile')
    parser.add_argument('--save', type=str, help='Set --save file_dir if want to save network.')
    parser.add_argument('--load', type=str, help='Set --load file_dir if want to load network.')
    parser.add_argument('--name', help='Name of task.')
    parser.add_argument('--dataset', help='Choose dataset: Minetto/icv15.', default='Minetto')
    parser.add_argument('--learnrate', type=str, help='Learning rate.',default="0.001")
    parser.add_argument('--epoch', type=str, help='Epoch size.',default="10")
    parser.add_argument('--lr_decay', help='Set --lr_decay to enbable learning rate decay.', action="store_true")
    # trajectory args
    parser.add_argument('--random', help='Set --random to replace trajectory with fully random generation.', action="store_true")
    parser.add_argument('--maxstep', type=int, help='Max lenth of single image, total lenth will be batch*maxstep.',default=3)
    parser.add_argument('--rotatev', type=float, help='Typical rotation speed (angle/FPS).',default=20)
    parser.add_argument('--shiftv', type=float, help='Typical shift speed (pixel/FPS).',default=20)
    parser.add_argument('--scalev', type=float, help='Typical scale speed (k/FPS).',default=0.3)

    args = parser.parse_args()
    args.dataset = args.dataset.lower() if(args.dataset)else args.dataset
    args.maxstep = max(args.maxstep,3)
    # args.debug = True
    # args.eval = True

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
            "\t Epoch size: {}.\n".format(cur_epoch)+\
            "\t Base network: {}.\n".format(args.basenet)+\
            "\t Optimizer: {}.\n".format(cur_opt)+\
            "\t LR decay: {}.\n".format('Yes' if(args.lr_decay)else 'No')+\
            "\t Dataset: {}.\n".format(cur_dataset)+\
            "\t Move step per-image: {}.\n".format(args.maxstep)+\
            "\t Init learning rate: {}.\n".format(cur_learnrate)+\
            "\t Taks name: {}.\n".format(args.name if(args.name)else 'None')+\
            "\t Load network: {}.\n".format(args.load if(args.load)else 'No')+\
            "\t Save network: {}.\n".format(args.save if(args.save)else 'No')+\
            "\t maxstep, rotatev, shiftv, scalev = {}, {}, {}, {}.\n".format(args.maxstep, args.rotatev, args.shiftv, args.scalev)+\
            "========\n"
        print(summarize)

        args.opt = cur_opt
        args.dataset = cur_dataset
        args.epoch = cur_epoch
        args.learnrate = cur_learnrate

        train(args)
        last_save = args.save