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
from torch.utils.tensorboard import SummaryWriter
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
from lib.utils.log_hlp import save_image
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

def train_siam(loader,net,opt,criteria,device,train_size,logger):
    maxh,maxw = 600,1200
    for batch,sample in enumerate(loader):
        assert('box' in sample and 'image' in sample)
        xs = sample['image']
        boxes_bth = sample['box']
        bxf = sample['box_format']
        batch_size = int(xs.shape[0])
        if('text' in sample):
            words_bth = sample['text']
        for i in range(batch_size):
            x = xs[i]
            img_size = x.shape[0:-1]
            x_nor = torch_img_normalize(x)
            if(x_nor.shape[-3]*x_nor.shape[-2]>maxh*maxw):
                x_nor = transform.resize(x_nor,(min(x_nor.shape[-3],maxh),min(x_nor.shape[-2],maxw),x_nor.shape[-1]),preserve_range=True)
                x_nor = torch.from_numpy(x_nor)
            x_nor = x_nor.reshape((1,x_nor.shape[0],x_nor.shape[1],x_nor.shape[2]))
            boxes = boxes_bth[i]
            if('poly' not in sample['box_format']):
                boxes = cv_box2cvbox(boxes,img_size,sample['box_format'])
            opt.zero_grad()
            x_nor = x_nor.float().permute(0,3,1,2).to(device)
            pred,feat = net(x_nor)
            sub_ch_loss = torch.zeros_like(pred[0,0,0,0])
            cnt = 0
            for box in boxes:
                # if(img_size!=x_nor.shape[1:-1]):
                #     box = np_box_resize(box,img_size,x_nor.shape[1:-1],'polyxy')
                sub_img,_ = cv_crop_image_by_bbox(
                    x.numpy(),np_polybox_minrect(box,'polyxy'),w_min=16*3,h_min=16*3)
                sub_img_nor = np_img_normalize(sub_img)
                sub_img_nor = np.expand_dims(sub_img_nor,0)
                sub_img_nor = torch.from_numpy(sub_img_nor).float().permute(0,3,1,2).to(device)
                try:
                    obj_map,obj_feat = net(sub_img_nor)
                    match_map,_ = net.match(obj_feat,feat)
                    sub_ch_mask, _,_,_ = cv_gen_gaussian(
                        np_box_resize(box,img_size,match_map.shape[2:],'polyxy'),
                        None,match_map.shape[2:],affin=False)
                    y = torch.from_numpy(np.expand_dims(sub_ch_mask,0)).to(match_map.device)
                    sub_ch_loss += criteria(match_map[:,0],y)
                    cnt+=1
                except:
                    continue
            if(sub_ch_loss==0.0):
                continue
            if(cnt):
                sub_ch_loss/=cnt
                try:
                    img = x.numpy().astype(np.uint8)
                    sub_img = sub_img.astype(np.uint8)
                    img[0:sub_img.shape[0],0:sub_img.shape[1],:]=sub_img
                    img = cv_draw_poly(img,np_box_resize(box,x.shape[:-1],img.shape[:-1],'polyxy'))

                    img_msk = cv_mask_image(img,match_map[0,0].to('cpu').detach().numpy())
                    logger.add_image('Tracking result', img_msk, batch*batch_size+i,dataformats='HWC')
                    img_msk = cv_mask_image(img,sub_ch_mask)
                    logger.add_image('Tracking GT', img_msk, batch*batch_size+i,dataformats='HWC')
                except:
                    None
            
            loss = sub_ch_loss
            logger.add_scalar('sub_ch_loss', sub_ch_loss.item(), batch*batch_size+i)
            logger.flush()
            print("sub_ch_loss {}, in step {}.".format(sub_ch_loss.item(),batch*batch_size+i))
            loss.backward()
            opt.step()
            del pred
            del feat
            del obj_map
            del obj_feat
            del match_map
            del y
            del x_nor
            del sub_img_nor
            del sample

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
    lod_dir = "/home/yomcoding/Pytorch/MyResearch/saved_model/craft_vgg_lstm_cp.pkl"
    teacher_pkl_dir = "/home/yomcoding/Pytorch/MyResearch/pre_train/craft_mlt_25k.pkl"
    sav_dir = "/home/yomcoding/Pytorch/MyResearch/saved_model/siam_craft_img.pkl"
    isdebug = True
    # use_net = 'craft_mob'
    use_dataset = 'minetto'
    # log_step_size = 1
    # use_cuda = False
    # num_workers=0
    # lr_decay_step_size = None

    dev = 'cuda' if(use_cuda)else 'cpu'
    # basenet = torch.load("/home/yomcoding/Pytorch/MyResearch/pre_train/craft_mlt_25k.pkl").float().to(dev)
    # net = SiameseCRAFT(base_net=basenet,feature_chs=32)
    # net = net.float().to(dev)
    net = torch.load('/home/yomcoding/Pytorch/MyResearch/saved_model/siam_craft2.pkl').float().to(dev)
    # if(os.path.exists(args.opt)):
    #     opt = torch.load(args.opt)
    # elif(args.opt.lower()=='adam'):
    #     opt = optim.Adam(net.parameters(), lr=lr, weight_decay=tcfg['OPT_DEC'])
    # elif(args.opt.lower() in ['adag','adagrad']):
    #     opt = optim.Adagrad(net.parameters(), lr=lr, weight_decay=tcfg['OPT_DEC'])
    # else:
    #     opt = optim.SGD(net.parameters(), lr=lr, momentum=tcfg['MMT'], weight_decay=tcfg['OPT_DEC'])
    opt = torch.load('/home/yomcoding/Pytorch/MyResearch/saved_model/siam_craft_opt2.pkl')
    # opt.add_param_group(net.parameters())
    # train_dataset = Total(
    #     os.path.join(__DEF_TTT_DIR,'Images','Train'),
    #     os.path.join(__DEF_TTT_DIR,'gt_pixel','Train'),
    #     os.path.join(__DEF_TTT_DIR,'gt_txt','Train'),
    #     out_box_format='polyxy',
    #     )
    train_dataset = ICDAR(
        os.path.join(__DEF_IC15_DIR,'images','train'),
        os.path.join(__DEF_IC15_DIR,'gt_txt','train'),
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
        "\t Optimizer: {}.\n".format(opt.__class__.__name__)+\
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
    logger = SummaryWriter(os.path.join(work_dir,'log','siam'))
    # logger = None

    # try:
    train_siam(dataloader,net,opt,loss,dev,len(train_dataset),logger)
    # except Exception as e:
    #     print(e)
        
    print("Saving model...")
    torch.save(net,sav_dir)
    print("Saving optimizer...")
    torch.save(opt,sav_dir+'_opt.pkl')

    time_usage = datetime.now()
    print("End at: {}.\n".format(time_usage.strftime("%Y%m%d-%H%M%S")))
    time_usage = time_usage - time_start
    print("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    pass
