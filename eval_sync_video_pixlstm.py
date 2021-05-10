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
import pandas as pd
import seaborn as sns
import pickle
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

import warnings
warnings.filterwarnings('ignore')

DEF_MOD_RATE = 0.3
DEF_WAVE_FUNC = lambda x: np.cos(2*x*np.pi)*DEF_MOD_RATE+1-DEF_MOD_RATE
DEF_FLUSH_COUNT = -1
DEF_LSTM_STATE_SIZE=(322,322)

@torch.no_grad()
def global_eval(model_list,maskch_list,model_name_list,maxstep:int=10,var=None,work_dir=None):
    rotatev = 30
    shiftv = 20
    scalev = 0.2

    image_size=(640, 640)
    time_start = datetime.now()
    time_cur = time_start

    if(work_dir and not os.path.exists(os.path.join(work_dir,'images'))):
        os.makedirs(os.path.join(work_dir,'images'))
    # logger = SummaryWriter(work_dir) if(writer)else None

    if(not model_name_list or len(model_name_list)<len(model_list)):
        if(not model_name_list):
            model_name_list=[]
        model_name_list += [model.__class__.__name__ for model in model_list[len(model_name_list):]]

    model_list = [model.cuda() for model in model_list]
    criterion_mask = MSE_2d_Loss(positive_mult=1,pixel_sum=False).cuda()
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
    all_eva_dict_no_blur = {o:defaultdict(list) for o in model_name_list}
    all_eva_dict_blur = {o:defaultdict(list) for o in model_name_list}
    all_score_dict = defaultdict(list)
    seed = np.random

    for stepi, sample in enumerate(eval_dataset):
        txts = sample['text']
        if(var):
            args={}
            if('rotate' in var):
                args['rotate']=var['rotate']*(seed.random()-0.5)*2.2
            if('shift' in var):
                args['shift']=(var['shift']*(seed.random()-0.5)*2.2,var['shift']*(seed.random()-0.5)*2.2)
            if('scale' in var):
                args['scale']=(var['scale']*((seed.random()-0.5)*0.4+1),var['scale']*((seed.random()-0.5)*0.4+1))

            image_list,poly_xy_list,Ms,step_state_dict = cv_gen_trajectory(sample['image'],maxstep,sample['box'],
                fluctuation=0,return_states=True,**args)
            blur_stepi = []

        else:
            rotate = maxstep * rotatev *(seed.random()-0.5)*2.2
            shiftx = maxstep * shiftv *(seed.random()-0.5)*2.2
            shifty = maxstep * shiftv *(seed.random()-0.5)*2.2
            scalex = maxstep * scalev *((seed.random()-0.5)*0.2+1)
            scaley = maxstep * scalev *((seed.random()-0.5)*0.2+1)
            image_list,poly_xy_list,Ms,step_state_dict,blur_stepi = cv_gen_trajectory(sample['image'],maxstep,sample['box'],
                rotate=rotate,shift=(shifty,shiftx),scale=(scaley,scalex),
                blur=True,blur_rate=0.6,blur_ksize=15,blur_intensity=0.2,
                blur_return_stepi=True,return_states=True,
                )
        
        x_sequence = torch.tensor(image_list,dtype = model_dtype, device=model_device)
        xnor = torch_img_normalize(x_sequence).permute(0,3,1,2)

        fgbxs_list,bgbxs_list = [],[]
        region_mask_np_list = []
        weight_mask_list = []
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
            if(bgbxs):
                weight_mask_list.append(cv_gen_binary_mask_by_poly(bgbxs,image_size,default_value=1,default_fill=0))
            else:
                weight_mask_list.append(np.ones(image_size,dtype=np.float32))

            fgbxs_list.append(np.array(fgbxs))
            bgbxs_list.append(np.array(bgbxs))

            all_fgbxs = np.array(all_fgbxs)
            region_mask_np = cv_gen_gaussian_by_poly(all_fgbxs,image_size)
            region_mask_np_list.append(region_mask_np)

        region_mask = torch.from_numpy(np.expand_dims(np.array(region_mask_np_list),1)).float().cuda()
        eva_dict = {o:defaultdict(list) for o in model_name_list}

        weight_mask = torch.from_numpy(np.expand_dims(np.array(weight_mask_list),-1)).to(model_dtype).to(model_device).permute(0,3,1,2)

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
                loss = criterion_mask(pred[:,maskch:maskch+1],region_mask[framei:framei+1],weight_mask[framei:framei+1])

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

            for i in range(len(recall_list)):
                all_eva_dict[model_name]['recall'].append(recall_list[i])
                all_eva_dict[model_name]['precision'].append(precision_list[i])
                all_eva_dict[model_name]['fscore'].append(fscore_list[i])
                all_eva_dict[model_name]['loss'].append(loss_list[i])
                for k,v in step_state_dict.items():
                    all_eva_dict[model_name][k].append(v[i])
                if(i in blur_stepi):
                    all_eva_dict_blur[model_name]['recall'].append(recall_list[i])
                    all_eva_dict_blur[model_name]['precision'].append(precision_list[i])
                    all_eva_dict_blur[model_name]['fscore'].append(fscore_list[i])
                    all_eva_dict_blur[model_name]['loss'].append(loss_list[i])
                    for k,v in step_state_dict.items():
                        all_eva_dict_blur[model_name][k].append(v[i])
                else:
                    all_eva_dict_no_blur[model_name]['recall'].append(recall_list[i])
                    all_eva_dict_no_blur[model_name]['precision'].append(precision_list[i])
                    all_eva_dict_no_blur[model_name]['fscore'].append(fscore_list[i])
                    all_eva_dict_no_blur[model_name]['loss'].append(loss_list[i])
                    for k,v in step_state_dict.items():
                        all_eva_dict_no_blur[model_name][k].append(v[i])
        if(work_dir):
            fig,axs = plt.subplots(
                nrows=len(model_list)+2, 
                ncols=1,
                # figsize=(2*len(image_list), 2*(2+len(model_list))),
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
        
                       
        # end of single sample
        # break

    return all_eva_dict,all_eva_dict_no_blur,all_eva_dict_blur

def log_plt(all_eva_dict,work_dir):
    log_name_list = ['recall_no_blur','precision_no_blur','fscore_no_blur','loss_no_blur','recall_blur','precision_blur','fscore_blur','loss_blur',]
    y_name = ['recall','fscore','loss']
    for model_name,eva_dict in all_eva_dict.items():
        # calculate performance difference
        all_ys_dict = defaultdict(list)
        all_xs_dict = defaultdict(list)
        y_fscore_all = []
        y_loss_all = []
        x_rotate_all = []
        x_shift_all = []
        x_scale_all = []
        if(y_name[0]+'_no_blur' in eva_dict and len(eva_dict[y_name[0]+'_no_blur'])>0):
            ys,ys_name = [],[]
            xs,xs_name = [],[]
            for log_name,log_v_list in eva_dict.items():
                itm_name = log_name.split('_')[0]
                if(itm_name in y_name):
                    all_ys_dict[itm_name]+=log_v_list
                    ys.append(np.array(log_v_list))
                else:
                    all_xs_dict[itm_name]+=log_v_list
            
            y_fscore_all+=all_eva_dict[model_name]['fscore_no_blur']
            y_loss_all+=all_eva_dict[model_name]['loss_no_blur']
            
            x_rotate_all+=all_eva_dict[model_name]['rotate_no_blur']
            x_shift_all+=[max(x,y) for x,y in zip(all_eva_dict[model_name]['shiftx_no_blur'],all_eva_dict[model_name]['shifty_no_blur'])]
            x_scale_all+=[max(x,y) for x,y in zip(all_eva_dict[model_name]['scalex_no_blur'],all_eva_dict[model_name]['scaley_no_blur'])]

            y_fscore = np.array(all_eva_dict[model_name]['fscore_no_blur'])
            y_loss = np.array(all_eva_dict[model_name]['loss_no_blur'])
            x_rotate = np.array(all_eva_dict[model_name]['rotate_no_blur'])
            x_shift = np.array([max(x,y) for x,y in zip(all_eva_dict[model_name]['shiftx_no_blur'],all_eva_dict[model_name]['shifty_no_blur'])])
            x_scale = np.array([max(x,y) for x,y in zip(all_eva_dict[model_name]['scalex_no_blur'],all_eva_dict[model_name]['scaley_no_blur'])])
            x_rotate_ind = np.argsort(x_rotate)
            x_shift_ind = np.argsort(x_shift)
            x_scale_ind = np.argsort(x_scale)
            fig,axs = plt.subplots(nrows=2, ncols=3,figsize=(2*2, 2*3),sharey=True)
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

            axs[0,0].plot(np.take(x_rotate,x_rotate_ind), np.take(y_fscore,x_rotate_ind), label=model_name)
            axs[0,1].plot(np.take(x_shift,x_shift_ind), np.take(y_fscore,x_shift_ind), label=model_name)
            axs[0,2].plot(np.take(x_scale,x_scale_ind), np.take(y_fscore,x_scale_ind), label=model_name)

            axs[1,0].plot(np.take(x_rotate,x_rotate_ind), np.take(y_loss,x_rotate_ind), label=model_name)
            axs[1,1].plot(np.take(x_shift,x_shift_ind), np.take(y_loss,x_shift_ind), label=model_name)
            axs[1,2].plot(np.take(x_scale,x_scale_ind), np.take(y_loss,x_scale_ind), label=model_name)
            for o in axs.reshape(-1):
                o.legend()
            fig.savefig(os.path.join(work_dir,'eva_no_blur.png'))

        if('recall_blur' in all_eva_dict[model_name] and all_eva_dict[model_name]['recall_blur']):
            y_fscore_all+=all_eva_dict[model_name]['fscore_blur']
            y_loss_all+=all_eva_dict[model_name]['loss_blur']
            x_rotate_all+=all_eva_dict[model_name]['rotate_blur']
            x_shift_all+=[max(x,y) for x,y in zip(all_eva_dict[model_name]['shiftx_blur'],all_eva_dict[model_name]['shifty_blur'])]
            x_scale_all+=[max(x,y) for x,y in zip(all_eva_dict[model_name]['scalex_blur'],all_eva_dict[model_name]['scaley_blur'])]
            y_fscore = np.array(all_eva_dict[model_name]['fscore_blur'])
            y_loss = np.array(all_eva_dict[model_name]['loss_blur'])
            x_rotate = np.array(all_eva_dict[model_name]['rotate_blur'])
            x_shift = np.array([max(x,y) for x,y in zip(all_eva_dict[model_name]['shiftx_blur'],all_eva_dict[model_name]['shifty_blur'])])
            x_scale = np.array([max(x,y) for x,y in zip(all_eva_dict[model_name]['scalex_blur'],all_eva_dict[model_name]['scaley_blur'])])
            x_rotate_ind = np.argsort(x_rotate)
            x_shift_ind = np.argsort(x_shift)
            x_scale_ind = np.argsort(x_scale)

            fig,axs = plt.subplots(nrows=2, ncols=3,figsize=(2*2, 2*3),sharey=True)
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

            axs[0,0].plot(np.take(x_rotate,x_rotate_ind), np.take(y_fscore,x_rotate_ind), label=model_name)
            axs[0,1].plot(np.take(x_shift,x_shift_ind), np.take(y_fscore,x_shift_ind), label=model_name)
            axs[0,2].plot(np.take(x_scale,x_scale_ind), np.take(y_fscore,x_scale_ind), label=model_name)
            axs[1,0].plot(np.take(x_rotate,x_rotate_ind), np.take(y_loss,x_rotate_ind), label=model_name)
            axs[1,1].plot(np.take(x_shift,x_shift_ind), np.take(y_loss,x_shift_ind), label=model_name)
            axs[1,2].plot(np.take(x_scale,x_scale_ind), np.take(y_loss,x_scale_ind), label=model_name)

            for o in axs.reshape(-1):
                o.legend()
            fig.savefig(os.path.join(work_dir,'eva_blur.png'))

        x_rotate_ind = np.argsort(x_rotate_all)
        x_shift_ind = np.argsort(x_shift_all)
        x_scale_ind = np.argsort(x_scale_all)

        fig,axs = plt.subplots(nrows=2, ncols=3,figsize=(2*2, 2*3),sharey=True)
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
        axs[0,0].plot(np.take(x_rotate_all,x_rotate_ind), np.take(y_fscore_all,x_rotate_ind), label=model_name)
        axs[0,1].plot(np.take(x_shift_all,x_shift_ind), np.take(y_fscore_all,x_shift_ind), label=model_name)
        axs[0,2].plot(np.take(x_scale_all,x_scale_ind), np.take(y_fscore_all,x_scale_ind), label=model_name)
        axs[1,0].plot(np.take(x_rotate_all,x_rotate_ind), np.take(y_loss_all,x_rotate_ind), label=model_name)
        axs[1,1].plot(np.take(x_shift_all,x_shift_ind), np.take(y_loss_all,x_shift_ind), label=model_name)
        axs[1,2].plot(np.take(x_scale_all,x_scale_ind), np.take(y_loss_all,x_scale_ind), label=model_name)

        for o in axs.reshape(-1):
            o.legend()
        fig.savefig(os.path.join(work_dir,'eva_all.png'))


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
    models,names,chs = [],[],[]
    pthdir = '/BACKUP/yom_backup/saved_model/lstm_ttt_Residual_region_pxnet.pth'
    maskch=0
    model_lstm = PIXLSTM_Residual(mask_ch=2,basenet='mobile',min_upc_ch=128,min_map_ch=32,
        include_final=False,pretrained=True).float()
    model_lstm.init_state(shape=DEF_LSTM_STATE_SIZE,batch_size=1)
    model_lstm.load_state_dict(copyStateDict(torch.load(pthdir)))
    models.append(model_lstm)
    names.append('2DLSTM')
    chs.append(0)

    pthdir = '/BACKUP/yom_backup/saved_model/CNN_ttt_region_pxnet.pth'
    maskch=0
    model_cnn = PIXCNN(mask_ch=2,basenet='mobile',min_upc_ch=128,min_map_ch=32,
        include_final=False,pretrained=True).float()
    model_cnn.load_state_dict(copyStateDict(torch.load(pthdir)))
    models.append(model_cnn)
    names.append('CNN')
    chs.append(0)

    pthdir = '/BACKUP/yom_backup/saved_model/org_ttt_region_pxnet.pth'
    maskch=0
    model_cnn_random = PIX_Unet(mask_ch=4,min_mask_ch=32,include_final=False,basenet='mobile',min_upc_ch=128,pretrained=False).float()
    model_cnn_random.load_state_dict(copyStateDict(torch.load(pthdir)))
    models.append(model_cnn_random)
    names.append('CNN')
    chs.append(2)

    # work_dir = os.path.join(DEF_WORK_DIR,'log','eva_on_ttt',datetime.now().strftime("%Y%m%d-%H%M%S"))
    work_dir = '/home/yomcoding/Pytorch/MyResearch/saved_model/'
    y_vars=[
        # "fscore", 
        "loss",
        ]
    x_vars = ['rotate','shiftx','shifty','scalex','scaley']

    fig,axs = plt.subplots(len(y_vars),len(x_vars),figsize=(4.2*len(x_vars),2.2*len(y_vars)),sharey=True)
    axs = axs.reshape(-1,len(x_vars))

    var = {'shift':300}
    x_vars_l = ['shiftx','shifty']
    all_eva_dict,all_eva_dict_no_blur,all_eva_dict_blur = global_eval(
        [model_lstm,model_cnn,model_cnn_random],[0,0,0],['LSTM','CNN','CNN_random'],var = var)

    with open(os.path.join(work_dir,'all_eva_shift.pkl'), 'wb') as pfile:
        pickle.dump(all_eva_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        
    for model_name, all_score_dict in all_eva_dict.items():
        plt_correlation(
            [all_score_dict[o] for o in x_vars_l],[all_score_dict[o] for o in y_vars],
            x_names=x_vars_l,y_names=y_vars,
            fig=fig,axs=np.stack([axs[:,i] for i,o in enumerate(x_vars) if(o in x_vars_l)],axis=-1),
            cur_label_name=model_name
            )

    all_eva_dict_df = defaultdict(list)
    for model_name, all_score_dict in all_eva_dict.items():
        _,single_list = next(iter(all_score_dict.items()))
        all_eva_dict_df['model_name'] += [model_name]*len(single_list)
        for k,vs in all_score_dict.items():
            all_eva_dict_df[k]+=vs

    all_eva_dict_df = pd.DataFrame.from_dict(all_eva_dict_df)
    sns_plot = sns.pairplot(
        all_eva_dict_df,
        hue='model_name',
        x_vars=x_vars_l,
        y_vars=y_vars,
        plot_kws={
            'marker':"+", 
            'linewidth':1,
            'alpha':0.8,
        },
    )
    sns_plot.savefig(os.path.join(work_dir,'all_eva_shift.png'))

    var = {'rotate':290}
    x_vars_l = ['rotate']
    all_eva_dict,all_eva_dict_no_blur,all_eva_dict_blur = global_eval(
        [model_lstm,model_cnn,model_cnn_random],[0,0,0],['LSTM','CNN','CNN_random'],var = var)

    with open(os.path.join(work_dir,'all_eva_rotate.pkl'), 'wb') as pfile:
        pickle.dump(all_eva_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)

    for model_name, all_score_dict in all_eva_dict.items():
        plt_correlation(
            [all_score_dict[o] for o in x_vars_l],[all_score_dict[o] for o in y_vars],
            x_names=x_vars_l,y_names=y_vars,
            fig=fig,axs=np.stack([axs[:,i] for i,o in enumerate(x_vars) if(o in x_vars_l)],axis=-1),
            cur_label_name=model_name
            )

    all_eva_dict_df = defaultdict(list)
    for model_name, all_score_dict in all_eva_dict.items():
        _,single_list = next(iter(all_score_dict.items()))
        all_eva_dict_df['model_name'] += [model_name]*len(single_list)
        for k,vs in all_score_dict.items():
            all_eva_dict_df[k]+=vs

    all_eva_dict_df = pd.DataFrame.from_dict(all_eva_dict_df)
    sns_plot = sns.pairplot(
        all_eva_dict_df,
        hue='model_name',
        x_vars=x_vars_l,
        y_vars=y_vars,
        plot_kws={
            'marker':"+", 
            'linewidth':1,
            'alpha':0.8,
        },
    )
    sns_plot.savefig(os.path.join(work_dir,'all_eva_rotate.png'))

    var = {'scale':2}
    x_vars_l = ['scalex','scaley']
    all_eva_dict,all_eva_dict_no_blur,all_eva_dict_blur = global_eval(
        [model_lstm,model_cnn,model_cnn_random],[0,0,0],['LSTM','CNN','CNN_random'],var = var)

    with open(os.path.join(work_dir,'all_eva_scale.pkl'), 'wb') as pfile:
        pickle.dump(all_eva_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)

    for model_name, all_score_dict in all_eva_dict.items():
        plt_correlation(
            [all_score_dict[o] for o in x_vars_l],[all_score_dict[o] for o in y_vars],
            x_names=x_vars_l,y_names=y_vars,
            fig=fig,axs=np.stack([axs[:,i] for i,o in enumerate(x_vars) if(o in x_vars_l)],axis=-1),
            cur_label_name=model_name
            )
    fig.savefig(os.path.join(work_dir,'all_eva_distributed.png'))
    all_eva_dict_df = defaultdict(list)
    for model_name, all_score_dict in all_eva_dict.items():
        _,single_list = next(iter(all_score_dict.items()))
        all_eva_dict_df['model_name'] += [model_name]*len(single_list)
        for k,vs in all_score_dict.items():
            all_eva_dict_df[k]+=vs

    all_eva_dict_df = pd.DataFrame.from_dict(all_eva_dict_df)
    sns_plot = sns.pairplot(
        all_eva_dict_df,
        hue='model_name',
        x_vars=x_vars_l,
        y_vars=y_vars,
        plot_kws={
            'marker':"+", 
            'linewidth':1,
            'alpha':0.8,
        },
    )
    sns_plot.savefig(os.path.join(work_dir,'all_eva_scale.png'))

    # all_eva_dict_df = defaultdict(list)
    # all_eva_dict,all_eva_dict_no_blur,all_eva_dict_blur = global_eval(
    #     [model_lstm,model_cnn,model_cnn_random],[0,0,0],['LSTM','CNN','CNN_random'])

    # for model_name, all_score_dict in all_eva_dict.items():
    #     _,single_list = next(iter(all_score_dict.items()))
    #     all_eva_dict_df['model_name'] += [model_name]*len(single_list)
    #     for k,vs in all_score_dict.items():
    #         all_eva_dict_df[k]+=vs

    # all_eva_dict_df = pd.DataFrame.from_dict(all_eva_dict_df)
    # sns_plot = sns.pairplot(
    #     all_eva_dict_df,
    #     hue='model_name',
    #     x_vars=x_vars,
    #     y_vars=y_vars,
    #     plot_kws={
    #         # 'marker':"+", 
    #         'linewidth':1,
    #         'alpha':0.8,
    #     },
    # )
    # sns_plot.savefig(os.path.join(work_dir,'all_eva.png'))
    # log_plt(ret,work_dir)
    print('end')