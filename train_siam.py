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
    
    for batch,sample in enumerate(loader):
        im_size = (sample['height'],sample['width'])
        pxy_dict = sample['gt']
        txt_dict = sample['txt']
        p_keys = list(pxy_dict.keys())
        p_keys.sort()
        vdo = sample['video']
        fm_cnt = 0

        # for character level tracking candidate
        wd_img_txt_box_dict = {} # dict[obj_id] = [img_tensor, word, box]
        obj_dict = {} # obj_dict[obj_id] = [start frame, end frame]
        for frame_id in p_keys:
            if(pxy_dict[frame_id].shape[0]==0):
                continue
            obj_ids = pxy_dict[frame_id][:,0].astype(np.int16)
            for oid in obj_ids:
                if(oid in obj_dict):
                    obj_dict[oid][-1] = frame_id
                else:
                    obj_dict[oid] = [frame_id,frame_id]
        

        while(vdo.isOpened()):
            ret, x = vdo.read()
            if(ret==False):
                break
            if(fm_cnt<p_keys[0] or fm_cnt not in pxy_dict or pxy_dict[fm_cnt].shape[0]==0):
                # skip the initial or non text frams
                fm_cnt+=1
                continue

            boxes = pxy_dict[fm_cnt][:,1:].reshape(-1,4,2)
            obj_ids = pxy_dict[fm_cnt][:,0].astype(np.int16)
            wrd_list = txt_dict[fm_cnt]
            rect_boxes = np_polybox_minrect(boxes,'polyxy')

            # x = cv2.resize(x,img_size,preserve_range=True)
            xnor = np_img_normalize(x)
            if(len(xnor.shape)==3): 
                xnor = np.expand_dims(xnor,0)
            xnor = torch.from_numpy(xnor).float().permute(0,3,1,2)
            opt.zero_grad()
            pred,feat = net(xnor.to(device))

            ch_mask, af_mask, ch_boxes_list, aff_boxes_list = cv_gen_gaussian(
                np_box_resize(boxes,x.shape[:-1],pred.shape[2:],'polyxy'),
                wrd_list,pred.shape[2:])

            ch_loss = criteria(pred[:,0],torch.from_numpy(np.expand_dims(ch_mask,0)).to(pred.device))
            af_loss = criteria(pred[:,1],torch.from_numpy(np.expand_dims(af_mask,0)).to(pred.device))
            
            # img = cv_mask_image(x,pred[0,0].to('cpu').detach().numpy())
            # if(logger):
            #     logger.add_image('Batch {} image'.format(batch), img, fm_cnt,dataformats='HWC')
            # tracking
            sub_ch_loss = torch.zeros_like(ch_loss)
            cnt = 0
            pops = []
            for o in wd_img_txt_box_dict:
                if(obj_dict[o][-1]<fm_cnt):
                    # if the object will not appear later
                    pops.append(o)
                    continue
                if(o not in obj_ids):
                    continue

                # numpy (h,w3)
                subx = wd_img_txt_box_dict[o][0]

                nid = np.where(obj_ids==o)[0][0]
                wd_old = wd_img_txt_box_dict[o][1]
                wd_now = wrd_list[nid]
                if(len(wd_old)<len(wd_now)):
                    # Need update 
                    # Simply delete old word and it will update later
                    pops.append(o)
                    continue
                elif(len(wd_old)>len(wd_now)+1):
                    # occlusion happen, cutting old image
                    try:
                        sp = wd_old.index(wd_now)
                        ep = sp+len(wd_now)
                        h,w = subx.shape[:-1]
                        sp/=len(wd_old)
                        ep/=len(wd_old)
                        if(w>=h):
                            sp = int(w*sp)
                            ep = int(w*ep)
                            subx = subx[:,sp:ep]
                        else:
                            sp = int(h*sp)
                            ep = int(h*ep)
                            subx = subx[sp:ep]
                    except:
                        continue
                if(subx.shape[-2]<8 or subx.shape[-3]<8):
                    continue
                
                nor_subx = np.expand_dims(np_img_normalize(subx),0)
                nor_subx = torch.from_numpy(nor_subx).float().permute(0,3,1,2).to(device)
                obj_map,obj_feat = net(nor_subx)
                match_map,_ = net.match(obj_feat,feat)
                trk_box = np_box_resize(boxes[nid],x.shape[:-1],match_map.shape[2:],'polyxy')
                sub_ch_mask, _,_,_ = cv_gen_gaussian(trk_box,None,match_map.shape[2:],affin=False)
                tracking_loss = criteria(match_map[:,0],torch.from_numpy(np.expand_dims(sub_ch_mask,0)).to(match_map.device))
                sub_ch_loss += tracking_loss

                np_subx = subx.astype(x.dtype)
                x[0:np_subx.shape[0],0:np_subx.shape[1],:]=np_subx
                img = cv_draw_poly(x,boxes[nid])
                img_msk = cv_mask_image(img,match_map[0,0].to('cpu').detach().numpy())
                if(logger):
                    logger.add_image('Batch {}, id {}, prediction'.format(batch,o), img_msk, fm_cnt,dataformats='HWC')
                img_msk = cv_mask_image(img,sub_ch_mask)
                if(logger):
                    logger.add_image('Batch {}, id {}, GT'.format(batch,o), img_msk, fm_cnt,dataformats='HWC')
                cnt+=1

            for o in pops:
                # delete old tracking target
                wd_img_txt_box_dict.pop(o)
            if(cnt>0):
                sub_ch_loss /= cnt
            loss = ch_loss + af_loss + sub_ch_loss
            
            if(logger):
                # logger.add_scalar('Batch {} ch_loss'.format(batch), ch_loss.item(), fm_cnt)
                # logger.add_scalar('Batch {} af_loss'.format(batch), af_loss.item(), fm_cnt)
                logger.add_scalar('Batch {} sub_ch_loss'.format(batch), sub_ch_loss.item(), fm_cnt)
                logger.add_scalar('Batch {} total loss'.format(batch), loss.item(), fm_cnt)
                logger.flush()
            print("Loss at batch {} step {}\t ch_loss={}\t af_loss={}\t sub_ch_loss={}\t\n".format(batch,fm_cnt,ch_loss.item(),af_loss.item(),sub_ch_loss.item()))
            
            loss.backward()
            opt.step()

            # AFTER tracking, update new objects
            for box,obj_id,wrd in zip(boxes,obj_ids,wrd_list):
                if(obj_id in wd_img_txt_box_dict):
                    continue
                sub_ch_img_np,_ = cv_crop_image_by_bbox(x,box,w_min=16*3,h_min=16*3)
                # sub_ch_img = cv_crop_image_by_bbox(x,box,32,32)
                # (1,3,h,w)
                # sub_ch_img = torch.from_numpy(np.expand_dims(sub_ch_img,0)).permute(0,3,1,2).float().to(device)
                wd_img_txt_box_dict[obj_id] = [sub_ch_img_np,wrd,box]

            # end of single frame
            fm_cnt+=1

    

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
    sav_dir = "/home/yomcoding/Pytorch/MyResearch/saved_model/siam_craft.pkl"
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
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=tcfg['OPT_DEC'])
    # elif(args.opt.lower() in ['adag','adagrad']):
    #     opt = optim.Adagrad(net.parameters(), lr=lr, weight_decay=tcfg['OPT_DEC'])
    # else:
    # opt = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    # opt = torch.load('/home/yomcoding/Pytorch/MyResearch/saved_model/siam_craft.pkl_opt.pkl')
    # opt.add_param_group(net.parameters())
    train_dataset = Minetto(__DEF_MINE_DIR)
    train_on_real = True
    x_input_function = None
    y_input_function = None
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
