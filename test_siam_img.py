import os
import sys
import platform
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
# =================Torch=======================
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import transform
import cv2
# =================Local=======================
from lib.model.craft import CRAFT,CRAFT_MOB,CRAFT_LSTM,CRAFT_MOTION,CRAFT_VGG_LSTM
from lib.model.siamfc import *
from lib.loss.mseloss import *
from lib.dataloader.total import Total
from lib.dataloader.icdar import ICDAR
from lib.dataloader.icdar_video import ICDARV
from lib.dataloader.minetto import Minetto
from lib.dataloader.base import BaseDataset
from lib.dataloader.synthtext import SynthText
from lib.utils.img_hlp import *
from lib.fr_craft import CRAFTTrainer
from lib.config.train_default import cfg as tcfg


__DEF_LOCAL_DIR = os.path.dirname(os.path.realpath(__file__))
__DEF_DATA_DIR = os.path.join(__DEF_LOCAL_DIR, 'dataset')
__DEF_CTW_DIR = os.path.join(__DEF_DATA_DIR, 'ctw')
__DEF_SVT_DIR = os.path.join(__DEF_DATA_DIR, 'svt', 'img')
__DEF_TTT_DIR = os.path.join(__DEF_DATA_DIR, 'totaltext')
__DEF_IC15_DIR = os.path.join(__DEF_DATA_DIR, 'ICDAR2015')
__DEF_IC19_DIR = os.path.join(__DEF_DATA_DIR, 'ICDAR2019')
__DEF_MSRA_DIR = os.path.join(__DEF_DATA_DIR, 'MSRA-TD500')
__DEF_ICV15_DIR = os.path.join(__DEF_DATA_DIR, 'ICDAR2015_video')
__DEF_MINE_DIR = os.path.join(__DEF_DATA_DIR, 'minetto')

if(platform.system().lower()[:7]=='windows'):__DEF_SYN_DIR = "D:\\development\\SynthText"
elif(os.path.exists("/BACKUP/yom_backup/SynthText")):__DEF_SYN_DIR = "/BACKUP/yom_backup/SynthText"
else:__DEF_SYN_DIR = os.path.join(__DEF_DATA_DIR, 'SynthText')

@torch.no_grad()
def train_siam(loader,net,criteria,device):
    net.eval()
    prec_lst,recal_lst = [],[]

    for batch,sample in enumerate(loader):
        prec,recal = 0.0,0.0
        trk_prec,trk_recal = 0.0,0.0
        assert('box' in sample and 'image' in sample)
        xs = sample['image']
        boxes_bth = sample['box']
        bxf = sample['box_format']
        batch_size = int(xs.shape[0])
        words_bth = None
        if('text' in sample):
            words_bth = sample['text']
        for i in range(batch_size):
            x = xs[i]
            gt_boxes = boxes_bth[i]

            img_size = x.shape[-3:-1]
            x_nor = torch_img_normalize(x)
            x_nor = x_nor.reshape((1,x_nor.shape[0],x_nor.shape[1],x_nor.shape[2]))
            x_nor = x_nor.float().permute(0,3,1,2).to(device)
            pred,feat = net(x_nor)

            # get numpy word mask 
            det_map = pred.detach().to('cpu').numpy()
            wd_map = np.where(det_map[0,0]>det_map[0,1],det_map[0,0],det_map[0,1])

            # detect box on mask
            boxes,label_map,labels = cv_get_box_from_mask(wd_map)
            matchs = cv_box_match(boxes,np_box_resize(gt_boxes,img_size,pred.shape[-2:],'polyxy'))
            mch_num = 0
            for mch in matchs:
                if(mch!=None and mch>0):
                    mch_num+=1

            prec += mch_num/boxes.shape[0]
            recal += mch_num/gt_boxes.shape[0]
            
            sig_trk_prec,sig_trk_recal = 0.0,0.0
            trcnt,totalcnt = 0,0
            gt_boxes_rect = np_polybox_minrect(gt_boxes,'polyxy')
            for boxi in range(gt_boxes_rect.shape[0]):
                sub_img,_ = cv_crop_image_by_bbox(x.numpy(),gt_boxes_rect[boxi])
                if(sub_img.shape[0]<16*3 or sub_img.shape[1]<16*3):
                    continue
                totalcnt+=1
                sub_img_nor = np.expand_dims(np_img_normalize(sub_img),0)
                sub_img_nor = torch.from_numpy(sub_img_nor).float().permute(0,2,3,1).to(pred.device)
                try:
                    obj_map,obj_feat = net(sub_img_nor)
                    match_map,_ = net.match(obj_feat,feat)
                    
                    boxes,label_map,labels = cv_get_box_from_mask(match_map[0,0].to('cpu').numpy())
                    ovlap = cv_box_overlap(boxes,np_box_resize(gt_boxes[boxi],img_size,match_map.shape[-2:],'polyxy'))
                    mxovlap = np.max(ovlap)
                    if(mxovlap>=0.5):
                        trcnt+=1
                        sig_trk_prec += 1/boxes.shape[0]
                        sig_trk_recal += 1
                    
                except:
                    continue
            sig_trk_prec /= totalcnt
            sig_trk_recal /= totalcnt
            print("successful tracking {}/{} boxes".format(trcnt,totalcnt))
            trk_prec+=sig_trk_prec
            trk_recal+=sig_trk_recal

        trk_prec /= batch_size
        trk_recal /= batch_size
        prec /= batch_size
        recal /= batch_size
        prec_lst.append(prec)
        recal_lst.append(recal)
        print("Detection precision {}, recall {}".format(prec,recal))
        print("Tracking precision {}, recall {}".format(trk_prec,trk_recal))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config trainer')

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

    args = parser.parse_args()
    use_net = args.net.lower()
    use_dataset = args.dataset.lower()
    time_start = datetime.now()
    isdebug = args.debug
    lod_dir = args.load
    teacher_pkl_dir = args.teacher
    lr = args.learnrate
    max_step = args.step if(not isdebug)else 1000
    use_cuda = True if(args.gpu>=0 and torch.cuda.is_available())else False
    lr_decay_step_size = tcfg['LR_DEC_STP']
    num_workers=4 if(platform.system().lower()[:7]!='windows')else 0
    batch = args.batch
    work_dir = "/BACKUP/yom_backup" if(platform.system().lower()[:7]!='windows' and os.path.exists("/BACKUP/yom_backup"))else __DEF_LOCAL_DIR
    log_step_size = args.logstp

    # For Debug config
    lod_dir = "/home/yomcoding/Pytorch/MyResearch/saved_model/siam_craft_img_init.pkl"
    teacher_pkl_dir = "/home/yomcoding/Pytorch/MyResearch/pre_train/craft_mlt_25k.pkl"
    sav_dir = "/home/yomcoding/Pytorch/MyResearch/saved_model/siam_craft.pkl"
    isdebug = True
    # use_net = 'craft_mob'
    use_dataset = 'minetto'
    # log_step_size = 1
    # use_cuda = False
    # num_workers=0
    # lr_decay_step_size = None

    dev = 'cuda' if(use_cuda)else 'cpu'
    net = torch.load(lod_dir).float().to(dev).eval()
    
    train_dataset = ICDAR(
        os.path.join(__DEF_IC15_DIR,'images','test'),
        os.path.join(__DEF_IC15_DIR,'gt_txt','test'),
        out_box_format='polyxy',)
    num_workers = 0
    batch = 1

    dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if(num_workers>0)else False,
        collate_fn=train_dataset.default_collate_fn,
        )

    loss = MSE_2d_Loss()

    summarize = "Start when {}.\n".format(time_start.strftime("%Y%m%d-%H%M%S")) +\
        "Working DIR: {}\n".format(work_dir)+\
        "Running with: \n"+\
        "\t Step size: {},\n\t Batch size: {}.\n".format(max_step,batch)+\
        "\t Input shape: x={},y={}.\n".format(args.datax,args.datay)+\
        "\t Network: {}.\n".format(net.__class__.__name__)+\
        "\t Dataset: {}.\n".format(train_dataset.__class__.__name__)+\
        "\t Init learning rate: {}.\n".format(lr)+\
        "\t Learning rate decay rate: {}.\n".format(tcfg['OPT_DEC'] if(tcfg['OPT_DEC']>0)else "Disabled")+\
        "\t Learning rate decay step: {}.\n".format(lr_decay_step_size if(lr_decay_step_size)else "Disabled")+\
        "\t Taks name: {}.\n".format(args.name if(args.name!=None)else net.__class__.__name__)+\
        "\t Teacher: {}.\n".format(teacher_pkl_dir)+\
        "\t Use GPU: {}.\n".format('Yes' if(use_cuda>=0)else 'No')+\
        "\t Load network: {}.\n".format(lod_dir if(lod_dir)else 'No')+\
        "\t Save network: {}.\n".format(args.save if(args.save)else 'No')+\
        "\t Is debug: {}.\n".format('Yes' if(isdebug)else 'No')+\
        ""
    print(summarize)

    try:
        train_siam(dataloader,net,loss,dev)
    except Exception as e:
        print(e)
        
    time_usage = datetime.now()
    print("End at: {}.\n".format(time_usage.strftime("%Y%m%d-%H%M%S")))
    time_usage = time_usage - time_start
    print("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    pass
