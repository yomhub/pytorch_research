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
    total_cnt = 0
    for stepi, sample in enumerate(eval_dataset):
        # if(total_cnt>10):
        #     break
        total_cnt+=1
        txts = sample['text']
        if(var):
            args={}
            total_step = var['total_step'] if('total_step' in var)else 10
            fluctuation = var['fluctuation'] if('fluctuation' in var)else 0.1
            if('rotate' in var):
                args['rotate']=var['rotate']
                if(fluctuation>0):
                    args['rotate'] *= (seed.random()-0.5)*(1+fluctuation)*2
            if('shift' in var):
                args['shift']=(var['shift'],var['shift'])
                if(fluctuation>0):
                    args['shift'] = (args['shift'][0]*(seed.random()-0.5)*(1+fluctuation)*2,args['shift'][1]*(seed.random()-0.5)*(1+fluctuation)*2)
            if('scale' in var):
                args['scale']=(var['scale'],var['scale']*((seed.random()-0.5)*0.4+1))
                if(fluctuation>0):
                    args['scale']=(args['scale'][0]*((seed.random()-0.5)*fluctuation*2+1),args['scale'][1]*((seed.random()-0.5)*fluctuation*2+1))
            if('blur_ksize' in var):
                args['blur_ksize'] = var['blur_ksize']
            if('blur_motion' in var):
                args['blur_motion'] = var['blur_motion']
            if('blur_stepi' in var):
                args['blur_stepi'] = var['blur_stepi']

            image_list,poly_xy_list,Ms,step_state_dict,blur_stepi = cv_gen_trajectory(sample['image'],total_step,sample['box'],
                fluctuation=0,blur_return_stepi=True,return_states=True,**args)

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

        for model,model_name,maskch in zip(model_list,model_name_list,maskch_list):
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

            eva_dict[model_name]['region_np']=region_np_list

            all_eva_dict[model_name]['recall'].append(recall_list)
            all_eva_dict[model_name]['precision'].append(precision_list)
            all_eva_dict[model_name]['fscore'].append(fscore_list)
            all_eva_dict[model_name]['loss'].append(loss_list)
            for k,v in step_state_dict.items():
                all_eva_dict[model_name][k].append(v)
            for i in range(len(image_list)):
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
                figsize=(1.2*len(image_list), 2*(2+len(model_list))),
                sharey=True)
            axs = axs.reshape(-1)
            for i in range(len(axs)):
                axs[i].get_yaxis().set_visible(False)
                axs[i].get_xaxis().set_visible(False)
            axs[0].set_title('Images',loc='left')
            axs[0].get_yaxis().label.set_text('Images')
            image_list_log = []
            boundary = np.array([[1,1],[image_size[0]-1,1],[image_size[0]-1,image_size[1]-1],[1,image_size[1]-1]])
            for i in range(len(image_list)):
                if(i in blur_stepi):
                    image_list_log.append(cv_draw_poly(image_list[i],boundary,color=(255,0,0),thickness=20))
                else:
                    image_list_log.append(image_list[i])
            axs[0].imshow(concatenate_images(image_list_log,line_wide=1))

            axs[1].set_title('GT',loc='left')
            axs[1].get_yaxis().label.set_text('GT')
            region_mask_rgb = [cv_heatmap(o) for o in region_mask_np_list]
            axs[1].imshow(concatenate_images(region_mask_rgb,line_wide=1))

            for modelid,model_name in enumerate(model_name_list):
                axs[2+modelid].get_yaxis().label.set_text(model_name)
                axs[2+modelid].set_title(model_name,loc='left')
                region_mask_rgb = [cv_heatmap(o,resize_to = image_size) for o in eva_dict[model_name]['region_np'][0]]
                axs[2+modelid].imshow(concatenate_images(region_mask_rgb,line_wide=1))

            fig.savefig(os.path.join(work_dir,'images','eva_{}.png'.format(sample['name'].split('.')[0])))
        
                       
        # end of single sample
        # break
    log_y = [
        'recall',
        'precision',
        'fscore',
    ]
    for model_name, all_score_dict in all_eva_dict.items():
        sys.stdout.write('====={}=====\n'.format(model_name))
        for k in log_y:
            v=np.mean(all_score_dict[k])
            sys.stdout.write('\t{}:{:4.3f}%\n'.format(k,v*100))
        sys.stdout.write('\n')
        sys.stdout.flush()

    return all_eva_dict,all_eva_dict_no_blur,all_eva_dict_blur

if __name__ == '__main__':
    eva_distribute = False
    eva_glob = False
    eva_blur = True
    models,names,chs = [],[],[]
    pthdir = '/BACKUP/yom_backup/saved_model/lstm_ttt_Residual_region_pxnet.pth'
    model_lstm = PIXLSTM_Residual(mask_ch=2,basenet='mobile',min_upc_ch=128,min_map_ch=32,
        include_final=False,pretrained=True).float()
    model_lstm.init_state(shape=DEF_LSTM_STATE_SIZE,batch_size=1)
    model_lstm.load_state_dict(copyStateDict(torch.load(pthdir)))
    models.append(model_lstm)
    names.append('2DLSTM')
    chs.append(0)

    pthdir = '/BACKUP/yom_backup/saved_model/CNN_ttt_region_pxnet.pth'
    model_cnn = PIXCNN(mask_ch=2,basenet='mobile',min_upc_ch=128,min_map_ch=32,
        include_final=False,pretrained=True).float()
    model_cnn.load_state_dict(copyStateDict(torch.load(pthdir)))
    models.append(model_cnn)
    names.append('CNN')
    chs.append(0)

    pthdir = '/BACKUP/yom_backup/saved_model/org_ttt_region_pxnet.pth'
    model_cnn_random = PIX_Unet(mask_ch=4,min_mask_ch=32,include_final=False,basenet='mobile',min_upc_ch=128,pretrained=False).float()
    model_cnn_random.load_state_dict(copyStateDict(torch.load(pthdir)))
    models.append(model_cnn_random)
    names.append('CNN_random')
    chs.append(2)

    # work_dir = os.path.join(DEF_WORK_DIR,'log','eva_on_ttt',datetime.now().strftime("%Y%m%d-%H%M%S"))
    work_dir = '/home/yomcoding/Pytorch/MyResearch/saved_model/'
    y_vars=[
        "fscore", 
        "loss",
        ]
    x_vars = ['rotate','shiftx','shifty','scalex','scaley']
    if(eva_distribute):
        fig,axs = plt.subplots(len(y_vars),len(x_vars),figsize=(4.2*len(x_vars),2.2*len(y_vars)),sharey=True)
        axs = axs.reshape(-1,len(x_vars))

        var = {'shift':300}
        x_vars_l = ['shiftx','shifty']
        all_eva_dict,all_eva_dict_no_blur,all_eva_dict_blur = global_eval(
            models,chs,names,var = var)

        with open(os.path.join(work_dir,'all_eva_shift.pkl'), 'wb') as pfile:
            pickle.dump(all_eva_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        for model_name, all_score_dict in all_eva_dict.items():
            plt_correlation(
                [np.array(all_score_dict[o]).reshape(-1) for o in x_vars_l],
                [np.array(all_score_dict[o]).reshape(-1) for o in y_vars],
                x_names=x_vars_l,y_names=y_vars,
                fig=fig,axs=np.stack([axs[:,i] for i,o in enumerate(x_vars) if(o in x_vars_l)],axis=-1),
                cur_label_name=model_name
                )

        all_eva_dict_df = {}
        for model_name, all_score_dict in all_eva_dict.items():
            _,single_list = next(iter(all_score_dict.items()))
            single_list = single_list[0]
            all_eva_dict_df['model_name'] = [model_name]*len(single_list)
            for k,vs in all_score_dict.items():
                all_eva_dict_df[k]=np.array(single).reshape(-1)

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
            models,chs,names,var = var)

        with open(os.path.join(work_dir,'all_eva_rotate.pkl'), 'wb') as pfile:
            pickle.dump(all_eva_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        for model_name, all_score_dict in all_eva_dict.items():
            plt_correlation(
                [np.array(all_score_dict[o]).reshape(-1) for o in x_vars_l],
                [np.array(all_score_dict[o]).reshape(-1) for o in y_vars],
                x_names=x_vars_l,y_names=y_vars,
                fig=fig,axs=np.stack([axs[:,i] for i,o in enumerate(x_vars) if(o in x_vars_l)],axis=-1),
                cur_label_name=model_name
                )

        all_eva_dict_df = {}
        for model_name, all_score_dict in all_eva_dict.items():
            _,single_list = next(iter(all_score_dict.items()))
            single_list = single_list[0]
            all_eva_dict_df['model_name'] = [model_name]*len(single_list)
            for k,vs in all_score_dict.items():
                all_eva_dict_df[k]=np.array(single).reshape(-1)

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
            models,chs,names,var = var)

        with open(os.path.join(work_dir,'all_eva_scale.pkl'), 'wb') as pfile:
            pickle.dump(all_eva_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        for model_name, all_score_dict in all_eva_dict.items():
            plt_correlation(
                [np.array(all_score_dict[o]).reshape(-1) for o in x_vars_l],
                [np.array(all_score_dict[o]).reshape(-1) for o in y_vars],
                x_names=x_vars_l,y_names=y_vars,
                fig=fig,axs=np.stack([axs[:,i] for i,o in enumerate(x_vars) if(o in x_vars_l)],axis=-1),
                cur_label_name=model_name
                )

        all_eva_dict_df = {}
        for model_name, all_score_dict in all_eva_dict.items():
            _,single_list = next(iter(all_score_dict.items()))
            single_list = single_list[0]
            all_eva_dict_df['model_name'] = [model_name]*len(single_list)
            for k,vs in all_score_dict.items():
                all_eva_dict_df[k]=np.array(single).reshape(-1)

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
        fig.savefig(os.path.join(work_dir,'all_eva_distributed.png'))

    if(eva_glob):
        y_vars=[
            "fscore", 
            "loss",
            ]
        x_vars = ['rotate','shiftx','shifty','scalex','scaley']
        all_eva_dict_df = defaultdict(list)
        all_eva_dict,all_eva_dict_no_blur,all_eva_dict_blur = global_eval(
            models,chs,names,work_dir=work_dir)

        with open(os.path.join(work_dir,'all_eva_global.pkl'), 'wb') as pfile:
            pickle.dump(all_eva_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        # for model_name, all_score_dict in all_eva_dict.items():
        #     plt_correlation(
        #         [all_score_dict[o] for o in x_vars],[all_score_dict[o] for o in y_vars],
        #         x_names=x_vars,y_names=y_vars,
        #         fig=fig,axs=np.stack([axs[:,i] for i,o in enumerate(x_vars) if(o in x_vars)],axis=-1),
        #         cur_label_name=model_name
        #         )

        for model_name, all_score_dict in all_eva_dict.items():
            _,single_list = next(iter(all_score_dict.items()))
            all_eva_dict_df['model_name'] += [model_name]*len(single_list)
            for k,vs in all_score_dict.items():
                all_eva_dict_df[k]+=vs

        all_eva_dict_df = pd.DataFrame.from_dict(all_eva_dict_df)
        sns_plot = sns.pairplot(
            all_eva_dict_df,
            hue='model_name',
            x_vars=x_vars,
            y_vars=y_vars,
            plot_kws={
                # 'marker':"+", 
                'linewidth':1,
                'alpha':0.8,
            },
        )
        sns_plot.savefig(os.path.join(work_dir,'all_eva.png'))

    if(eva_blur):
        base_rotate = 15
        total_step = 5
        y_vars=["loss"]
        x_vars=['rotate']
        blur_ksize_list = [15,25,35]
        all_eva_dict_list,all_eva_dict_no_blur_list,all_eva_dict_blur_list = [],[],[]
        for blur_ksize in blur_ksize_list:
            var = {
                'rotate':base_rotate*total_step,
                'blur_motion':True,
                'blur_ksize':blur_ksize,
                'blur_stepi':[1,3]
                }
            all_eva_dict,all_eva_dict_no_blur,all_eva_dict_blur = global_eval(
                models,chs,names,work_dir=None)

            with open(os.path.join(work_dir,'all_eva_dict_k{}.pkl'.format(blur_ksize)), 'wb') as pfile:
                pickle.dump(all_eva_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(work_dir,'all_eva_dict_no_blur_k{}.pkl'.format(blur_ksize)), 'wb') as pfile:
                pickle.dump(all_eva_dict_no_blur, pfile, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(work_dir,'all_eva_dict_blur_k{}.pkl'.format(blur_ksize)), 'wb') as pfile:
                pickle.dump(all_eva_dict_blur, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        for model_name in names:
            all_eva_dict_df = defaultdict(list)
            for blur_ksize,all_eva_dict_blur in zip(blur_ksize_list,all_eva_dict_blur_list):
                all_score_dict = all_eva_dict_blur['model_name']
                len_single_list = 0
                for k,vs in all_score_dict.items():
                    if(k in y_vars or k in x_vars):
                        o = np.array(vs).reshape(-1).tolist()
                        all_eva_dict_df[k]+=o
                        len_single_list = len(o)
                all_eva_dict_df['blur_ksize'] += ['k:{}'.format(blur_ksize)]*len_single_list
            all_eva_dict_df = pd.DataFrame.from_dict(all_eva_dict_df)

            sns_plot = sns.pairplot(
                all_eva_dict_df,
                hue='blur_ksize',
                x_vars=x_vars,
                y_vars=y_vars,
                plot_kws={
                    # 'marker':"+", 
                    'linewidth':1,
                    'alpha':0.8,
                },
            )
            sns_plot.savefig(os.path.join(work_dir,'{}_blur.png'.format(model_name)))
    print('end')