import os
import sys
import platform
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from shapely.geometry import Polygon
# =================Torch=======================
import torch
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# =================Local=======================
from lib.dataloader.icdar_video import ICDARV
from lib.dataloader.total import Total
from lib.model.pixel_map import PIXLSTM_Residual,PIX_Unet,PIXCNN
from lib.loss.mseloss import *
from lib.utils.img_hlp import *
from lib.utils.log_hlp import *
from dirs import *

DEF_MOD_RATE = 0.3
DEF_WAVE_FUNC = lambda x: np.cos(2*x*np.pi)*DEF_MOD_RATE+1-DEF_MOD_RATE
DEF_FLUSH_COUNT = -1
DEF_LSTM_STATE_SIZE=(322,322)

@torch.no_grad()
def myeval(model_list,maskch_list,model_name_list,maxstep:int=10,writer:bool=True):
    rotatev = 30
    shiftv = 20
    scalev = 0.2

    image_size=(640, 640)
    time_start = datetime.now()
    time_cur = time_start
    work_dir = os.path.join(DEF_WORK_DIR,'log','eva_on_ttt',time_start.strftime("%Y%m%d-%H%M%S"))
    if(not os.path.exists(os.path.join(work_dir,'images'))):
        os.makedirs(os.path.join(work_dir,'images'))
    # logger = SummaryWriter(work_dir) if(writer)else None

    if(not model_name_list or len(model_name_list)<len(model_list)):
        if(not model_name_list):
            model_name_list=[]
        model_name_list += [model.__class__.__name__ for model in model_list[len(model_name_list):]]

    model_list = [model.cuda() for model in model_list]
    criterion_mask = MSE_2d_Loss(pixel_sum=False).cuda()
    _,d = next(iter(model_list[0].state_dict().items()))
    model_device,model_dtype = d.device,d.dtype

    # eval_dataset = Minetto(DEF_MINE_DIR)
    eval_dataset = Total(
        os.path.join(DEF_TTT_DIR,'images','test'),
        os.path.join(DEF_TTT_DIR,'gt_pixel','test'),
        os.path.join(DEF_TTT_DIR,'gt_txt','test'),
        image_size=image_size,
        include_bg=True,
        )
    
    all_eva_dict = {o:defaultdict(list) for o in model_name_list}
    all_score_dict = defaultdict(list)
    log_name_list = ['recall_no_blur','precision_no_blur','fscore_no_blur','loss_no_blur','recall_blur','precision_blur','fscore_blur','loss_blur',]

    for stepi, sample in enumerate(eval_dataset):
        txts = sample['text']
        seed = np.random
        rotate = maxstep * rotatev *(seed.random()-0.5)*2.2
        shiftx = maxstep * shiftv *(seed.random()-0.5)*2.2
        shifty = maxstep * shiftv *(seed.random()-0.5)*2.2
        scalex = maxstep * scalev *(seed.random()-0.5)*2.2
        scaley = maxstep * scalev *(seed.random()-0.5)*2.2
        image_list,poly_xy_list,Ms,blur_stepi = cv_gen_trajectory(sample['image'],maxstep,sample['box'],
            rotate=rotate,shift=(shifty,shiftx),scale=(scaley,scalex)
            ,blur=True,blur_rate=0.6,blur_ksize=15,blur_intensity=0.2,
            blur_return_stepi=True,
            )

        x_sequence = torch.tensor(image_list,dtype = model_dtype, device=model_device)
        xnor = torch_img_normalize(x_sequence).permute(0,3,1,2)

        fgbxs_list,bgbxs_list = [],[]
        region_mask_np_list = []
        # generate fg/bg in each frame
        for framei in range(len(image_list)):
            boxes = poly_xy_list[framei]
            fgbxs,bgbxs,all_fgbxs = [],[],[]

            # devide boxes into fg,bg
            for o in boxes:
                if(txts!='#'):
                    all_fgbxs.append(o)
                    part = Polygon(np.clip(o,0,min(image_size)))
                    if(part.is_valid):
                        hole = Polygon(o)
                        if(hole.intersects(part) and hole.intersection(part).area>max(10,hole.area/3)):
                            fgbxs.append(o)
                            continue
                    else:
                        continue
                bgbxs.append(o)

            fgbxs_list.append(np.array(fgbxs))
            bgbxs_list.append(np.array(bgbxs))

            all_fgbxs = np.array(all_fgbxs)
            region_mask_np = cv_gen_gaussian_by_poly(all_fgbxs,image_size)
            region_mask_np_list.append(region_mask_np)

        region_mask = torch.from_numpy(np.expand_dims(np.array(region_mask_np_list),1)).float().cuda()
        eva_dict = {o:defaultdict(list) for o in model_name_list}

        for model,model_name in zip(model_list,model_name_list):
            try:
                model.lstmc.zero_()
                model.lstmh.zero_()
            except:
                None

            recall_list,precision_list,fscore_list,loss_list = [],[],[],[]
            region_np_list = []
            for framei in range(len(image_list)):

                pred,feat = model(xnor[framei:framei+1])
                loss = criterion_mask(pred[:,maskch:maskch+1],region_mask[framei:framei+1])

                region_np = pred[0,maskch].cpu().detach().numpy()
                region_np[region_np<0.2]=0.0
                region_np_list.append(region_np)

                det_boxes, label_mask, label_list = cv_get_box_from_mask(region_np,region_mean_split=True)
                if(det_boxes.shape[0]>0):
                    det_boxes = np_box_resize(det_boxes,region_np.shape[-2:],x_sequence.shape[-3:-1],'polyxy')
                    ids,mask_precision,mask_recall = cv_box_match(det_boxes,fgbxs,bgbxs,ovth=0.5)
                else:
                    mask_precision,mask_recall=0.0,0.0

                mask_fscore = calculate_fscore(mask_precision,mask_recall)

                recall_list.append(mask_recall)
                precision_list.append(mask_precision)
                fscore_list.append(mask_fscore)
                loss_list.append(loss.item())

            eva_dict[model_name]['recall'].append(np.array(recall_list))
            eva_dict[model_name]['precision'].append(np.array(precision_list))
            eva_dict[model_name]['fscore'].append(np.array(fscore_list))
            eva_dict[model_name]['loss'].append(np.array(loss_list))
            eva_dict[model_name]['region_np'].append(region_np_list)

        if(writer):
            
            fig,axs = plt.subplots(
                nrows=len(model_list)+2, 
                ncols=1,
                figsize=(2*len(image_list), 2*(2+len(model_list))),
                sharey=True)
            axs = axs.reshape(-1)
            for i in range(len(axs)):
                axs[i].get_yaxis().set_visible(False)
                axs[i].get_xaxis().set_visible(False)

            axs[0].get_yaxis().label.set_text('Images')
            image_list_log = []
            boundary = np.array([[1,1],[image_size[0]-1,1],[image_size[0]-1,image_size[1]-1],[1,image_size[1]-1]])
            for i in range(len(image_list)):
                if(i in blur_stepi):
                    image_list_log.append(cv_draw_poly(image_list[i],boundary,color=(255,0,0),thickness=9))
                else:
                    image_list_log.append(image_list[i])
            axs[0].imshow(concatenate_images(image_list_log,line_wide=1))

            axs[1].get_yaxis().label.set_text('GT')
            region_mask_rgb = [cv_heatmap(o) for o in region_mask_np_list]
            axs[1].imshow(concatenate_images(region_mask_rgb,line_wide=1))

            for modelid in range(len(model_list)):
                axs[2+modelid].get_yaxis().label.set_text(model_name_list[modelid])
                region_mask_rgb = [cv_heatmap(o,resize_to = image_size) for o in eva_dict[model_name]['region_np'][0]]
                axs[2+modelid].imshow(concatenate_images(region_mask_rgb,line_wide=1))

            fig.savefig(os.path.join(work_dir,'images','eva_{}_.png'.format(sample['name'].split('.')[0])))
        
        for model_name in model_name_list:
            recall = eva_dict[model_name]['recall']
            precision = eva_dict[model_name]['precision']
            fscore = eva_dict[model_name]['fscore']
            loss = eva_dict[model_name]['loss']

            recall_blur, recall_no_blur = [],[]
            precision_blur, precision_no_blur = [],[]
            fscore_blur, fscore_no_blur = [],[]
            loss_blur, loss_no_blur = [],[]
            for i in range(len(recall)):
                if(i in blur_stepi):
                    recall_blur.append(recall[i])
                    precision_blur.append(precision[i])
                    fscore_blur.append(fscore[i])
                    loss_blur.append(loss[i])
                else:
                    recall_no_blur.append(recall[i])
                    precision_no_blur.append(precision[i])
                    fscore_no_blur.append(fscore[i])
                    loss_no_blur.append(loss[i])

            if(recall_no_blur):
                recall_no_blur = np.array(recall_no_blur)
                precision_no_blur = np.array(precision_no_blur)
                fscore_no_blur = np.array(fscore_no_blur)
                loss_no_blur = np.array(loss_no_blur)
                mean_recall_no_blur = np.mean(recall_no_blur)
                mean_precision_no_blur = np.mean(precision_no_blur)
                mean_fscore_no_blur = np.mean(fscore_no_blur)
                mean_loss_no_blur = np.mean(loss_no_blur)
    
                all_eva_dict[model_name]['recall_no_blur'].append(mean_recall_no_blur)
                all_eva_dict[model_name]['precision_no_blur'].append(mean_precision_no_blur)
                all_eva_dict[model_name]['fscore_no_blur'].append(mean_fscore_no_blur)
                all_eva_dict[model_name]['loss_no_blur'].append(mean_loss_no_blur)
                all_eva_dict[model_name]['rotate_no_blur'].append(rotate)
                all_eva_dict[model_name]['shiftx_no_blur'].append(shiftx)
                all_eva_dict[model_name]['shifty_no_blur'].append(shifty)
                all_eva_dict[model_name]['scalex_no_blur'].append(scalex)
                all_eva_dict[model_name]['scaley_no_blur'].append(scaley)

            if(recall_blur):
                all_eva_dict[model_name]['recall_blur'].append(np.mean(recall_blur))
                all_eva_dict[model_name]['precision_blur'].append(np.mean(precision_blur))
                all_eva_dict[model_name]['fscore_blur'].append(np.mean(fscore_blur))
                all_eva_dict[model_name]['loss_blur'].append(np.mean(loss_blur))
                all_eva_dict[model_name]['rotate_blur'].append(rotate)
                all_eva_dict[model_name]['shiftx_blur'].append(shiftx)
                all_eva_dict[model_name]['shifty_blur'].append(shifty)
                all_eva_dict[model_name]['scalex_blur'].append(scalex)
                all_eva_dict[model_name]['scaley_blur'].append(scaley)
            
        # end of single sample
        # break

        fig,axs = plt.subplots(nrows=2, ncols=3,figsize=(len(image_list), 2*(2+len(model_list))),sharey=True)
        for o in axs[:,0]:
            o.get_xaxis().label.set_text('Rotation')
        for o in axs[:,1]:
            o.get_xaxis().label.set_text('Shift')
        for o in axs[:,2]:
            o.get_xaxis().label.set_text('Scale')
        for o in axs[0,:]:
            o.get_yaxis().label.set_text('F-score')
        for o in axs[1,:]:
            o.get_yaxis().label.set_text('Loss')

        for model_id,model_name in enumerate(model_name_list):
            # calculate performance difference
            if('recall_no_blur' in all_eva_dict[model_name] and all_eva_dict[model_name]['recall_no_blur']):
                ax = axs[0,0]
                x_fscore = np.array(all_eva_dict[model_name]['fscore_no_blur'])
                x_loss = np.array(all_eva_dict[model_name]['loss_no_blur'])
                y_rotate = np.array(all_eva_dict[model_name]['rotate_no_blur'])
                y_shift = np.array([max(x,y) for x,y in zip(all_eva_dict[model_name]['shiftx_no_blur'],all_eva_dict[model_name]['shifty_no_blur'])])
                y_scale = np.array([max(x,y) for x,y in zip(all_eva_dict[model_name]['scalex_no_blur'],all_eva_dict[model_name]['scaley_no_blur'])])
                y_rotate_ind = np.argsort(y_rotate)
                y_shift_ind = np.argsort(y_shift)
                y_scale_ind = np.argsort(y_scale)

                axs[0,0].plot(np.take(x_fscore,y_rotate_ind),np.take(y_rotate,y_rotate_ind), label=model_name+'_no_blur')
                axs[0,1].plot(np.take(x_fscore,y_shift_ind),np.take(y_shift,y_shift_ind), label=model_name+'_no_blur')
                axs[0,2].plot(np.take(x_fscore,y_scale_ind),np.take(y_scale,y_scale_ind), label=model_name+'_no_blur')

                axs[1,0].plot(np.take(x_loss,y_rotate_ind),np.take(y_rotate,y_rotate_ind), label=model_name+'_no_blur')
                axs[1,1].plot(np.take(x_loss,y_shift_ind),np.take(y_shift,y_shift_ind), label=model_name+'_no_blur')
                axs[1,2].plot(np.take(x_loss,y_scale_ind),np.take(y_scale,y_scale_ind), label=model_name+'_no_blur')

            if('recall_blur' in all_eva_dict[model_name] and all_eva_dict[model_name]['recall_blur']):
                ax = axs[0,0]
                x_fscore = np.array(all_eva_dict[model_name]['fscore_blur'])
                x_loss = np.array(all_eva_dict[model_name]['loss_blur'])
                y_rotate = np.array(all_eva_dict[model_name]['rotate_blur'])
                y_shift = np.array([max(x,y) for x,y in zip(all_eva_dict[model_name]['shiftx_blur'],all_eva_dict[model_name]['shifty_blur'])])
                y_scale = np.array([max(x,y) for x,y in zip(all_eva_dict[model_name]['scalex_blur'],all_eva_dict[model_name]['scaley_blur'])])
                y_rotate_ind = np.argsort(y_rotate)
                y_shift_ind = np.argsort(y_shift)
                y_scale_ind = np.argsort(y_scale)

                axs[0,0].plot(np.take(x_fscore,y_rotate_ind),np.take(y_rotate,y_rotate_ind), label=model_name+'_blur')
                axs[0,1].plot(np.take(x_fscore,y_shift_ind),np.take(y_shift,y_shift_ind), label=model_name+'_blur')
                axs[0,2].plot(np.take(x_fscore,y_scale_ind),np.take(y_scale,y_scale_ind), label=model_name+'_blur')

                axs[1,0].plot(np.take(x_loss,y_rotate_ind),np.take(y_rotate,y_rotate_ind), label=model_name+'_blur')
                axs[1,1].plot(np.take(x_loss,y_shift_ind),np.take(y_shift,y_shift_ind), label=model_name+'_blur')
                axs[1,2].plot(np.take(x_loss,y_scale_ind),np.take(y_scale,y_scale_ind), label=model_name+'_blur')
        
        for o in axs.reshape(-1):
            o.legend()
        s = ''
        for o in model_name_list:
            s+='_'+o
        fig.savefig(os.path.join(work_dir,'eva{}_.png'.format(s)))
        # if(logger):
        #     region_np = pred[0,maskch].detach().cpu().numpy()
        #     img = x
        #     img = cv_draw_poly(img,fgbxs,'GT',color=(255,0,0))
        #     img = cv_draw_poly(img,det_boxes,'Pred',color=(0,255,0))

        #     logger.add_image('X|GT|Pred in {}'.format(sample['name']),concatenate_images([img,cv_heatmap(region_mask_np),cv_heatmap(region_np)]),fm_cnt,dataformats='HWC')
        #     logger.add_scalar('Loss/ {}'.format(sample['name']),loss.item(),fm_cnt)
        #     logger.add_scalar('Precision/ {}'.format(sample['name']),mask_precision,fm_cnt)
        #     logger.add_scalar('Recall/ {}'.format(sample['name']),mask_recall,fm_cnt)
        #     logger.add_scalar('F-score/ {}'.format(sample['name']),mask_fscore,fm_cnt)
        #     logger.flush()
        # fm_cnt += 1
    
    for model_name, all_score_dict in all_eva_dict.items():
        sys.stdout.write("============================\n")
        sys.stdout.write("Model: {}\n".format(model_name))
        for k in log_name_list:
            if(k in all_score_dict):
                v = all_score_dict[k]
                sys.stdout.write("\t {}: Mean {:3.3f}%, variance {:3.3f}.\n".format(k,np.mean(v)*100,np.var(v)))
        sys.stdout.write("\n")

    sys.stdout.flush()

    return 0

if __name__ == '__main__':
    pthdir = '/BACKUP/yom_backup/saved_model/lstm_ttt_Residual_region_pxnet.pth'
    maskch=0
    model_lstm = PIXLSTM_Residual(mask_ch=2,basenet='mobile',min_upc_ch=128,min_map_ch=32,
        include_final=False,pretrained=True).float()
    model_lstm.init_state(shape=DEF_LSTM_STATE_SIZE,batch_size=1)
    model_lstm.load_state_dict(copyStateDict(torch.load(pthdir)))

    pthdir = '/BACKUP/yom_backup/saved_model/CNN_ttt_region_pxnet.pth'
    maskch=0
    model_cnn = PIXCNN(mask_ch=2,basenet='mobile',min_upc_ch=128,min_map_ch=32,
        include_final=False,pretrained=True).float()
    model_cnn.load_state_dict(copyStateDict(torch.load(pthdir)))

    myeval([model_lstm,model_cnn],[0,0],['LSTM','CNN'],writer=False)
