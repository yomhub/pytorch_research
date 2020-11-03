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
from lib.dataloader.icdar import *
from lib.dataloader.icdar_video import ICDARV
from lib.dataloader.minetto import Minetto
from lib.dataloader.base import BaseDataset
from lib.dataloader.synthtext import SynthText
from lib.utils.img_hlp import *
from lib.utils.log_hlp import save_image
from lib.fr_craft import CRAFTTrainer
from lib.config.train_default import cfg as tcfg
from dirs import *


def train_siam(loader,net,opt,criteria,train_size,logger,work_dir,train_detector=False):
    maxh,maxw = 600,1200
    max_boxes = 3
    for o in net.state_dict().values():
        device = o.device
        break
    for batch,sample in enumerate(loader):
        assert('box' in sample and 'image' in sample)
        xs = sample['image']
        boxes_bth = sample['box']
        bxf = sample['box_format']
        
        batch_size = int(xs.shape[0])
        words_bth = None
        if('text' in sample):
            words_bth = sample['text']
        for i in range(batch_size):
            # try:
            if(1):
                x = xs[i]
                wrd_list = sample['text'][i]
                img_size = x.shape[0:-1]
                x_nor = torch_img_normalize(x)
                if(x_nor.shape[-3]*x_nor.shape[-2]>maxh*maxw):
                    x_nor = transform.resize(x_nor,(min(x_nor.shape[-3],maxh),min(x_nor.shape[-2],maxw),x_nor.shape[-1]),preserve_range=True)
                    x_nor = torch.from_numpy(x_nor)
                x_nor = x_nor.reshape((1,x_nor.shape[0],x_nor.shape[1],x_nor.shape[2]))
                boxes = boxes_bth[i]
                
                if('poly' not in sample['box_format']):
                    boxes = cv_box2cvbox(boxes,img_size,sample['box_format'])

                loss_dict = {}
                opt.zero_grad()
                x_nor = x_nor.float().permute(0,3,1,2).to(device)
                pred,feat = net(x_nor)
                if(train_detector):
                    resized_boxes = np_box_resize(boxes,img_size,pred.shape[2:],'polyxy')
                    resized_recbox = np_polybox_minrect(resized_boxes)
                    pred_ch_mask = pred[0,0].to('cpu').detach().numpy()
                    pred_af_mask = pred[0,1].to('cpu').detach().numpy()
                    ch_mask, af_mask, ch_boxes_list, aff_boxes_list = cv_gen_gaussian(resized_boxes,wrd_list,pred.shape[2:])
                    for bxi in range(resized_recbox.shape[0]):
                        spx,spy = resized_recbox[bxi][0].astype(np.int16)
                        sub_ch_mask,_ = cv_crop_image_by_bbox(pred_ch_mask,resized_recbox[bxi])
                        sub_af_mask,_ = cv_crop_image_by_bbox(pred_af_mask,resized_recbox[bxi])
                        subh,subw = sub_ch_mask.shape
                        nLabels, labels = cv2.connectedComponents(
                            np.logical_and(sub_ch_mask>=0.4,sub_ch_mask>sub_af_mask).astype(np.uint8),
                            connectivity=4)
                        if(nLabels<len(wrd_list[bxi])//2):
                            ch_mask[spy:subh+spy,spx:subw+spx] = pred_ch_mask[spy:subh+spy,spx:subw+spx]
                            af_mask[spy:subh+spy,spx:subw+spx] = pred_af_mask[spy:subh+spy,spx:subw+spx]

                    if(work_dir):
                        img_msk = cv_mask_image(x.numpy(),ch_mask)
                        save_image(os.path.join(work_dir,'images','gt_ch_mask_'+sample['name'][i]),img_msk)
                        img_msk = cv_mask_image(x.numpy(),pred_ch_mask)
                        save_image(os.path.join(work_dir,'images','pred_ch_mask_'+sample['name'][i]),img_msk)
                        img_msk = cv_mask_image(x.numpy(),af_mask)
                        save_image(os.path.join(work_dir,'images','gt_af_mask_'+sample['name'][i]),img_msk)
                        img_msk = cv_mask_image(x.numpy(),pred_af_mask)
                        save_image(os.path.join(work_dir,'images','pred_af_mask_'+sample['name'][i]),img_msk)

                    loss_dict['ch_loss'] = criteria(pred[:,0],torch.from_numpy(np.expand_dims(ch_mask,0)).to(pred.device))
                    loss_dict['af_loss'] = criteria(pred[:,1],torch.from_numpy(np.expand_dims(af_mask,0)).to(pred.device))
                loss_dict['sub_ch_loss'] = 0.0
                cnt = 0
                
                # Select top max_boxes biggest boxes 
                recbox = np_polybox_minrect(boxes,'polyxy')
                ws = np.linalg.norm(recbox[:,0]-recbox[:,1],axis=-1)
                hs = np.linalg.norm(recbox[:,0]-recbox[:,3],axis=-1)
                inds = np.argsort(ws*hs)[::-1]
                boxes = boxes[inds[:max_boxes]]
                recbox = recbox[inds[:max_boxes]]
                for bxi,box in enumerate(boxes):
                    # if(img_size!=x_nor.shape[1:-1]):
                    #     box = np_box_resize(box,img_size,x_nor.shape[1:-1],'polyxy')
                    sub_img,_ = cv_crop_image_by_polygon(
                        x.numpy(),boxes[bxi],w_min=16*3,h_min=16*3)
                    sub_img_nor = np_img_normalize(sub_img)
                    sub_img_nor = np.expand_dims(sub_img_nor,0)
                    sub_img_nor = torch.from_numpy(sub_img_nor).float().permute(0,3,1,2).to(device)
                    try:
                        obj_map,obj_feat = net(sub_img_nor)
                        # obj_feat*=obj_map[:,0].reshape(obj_feat.shape[0],1,obj_feat.shape[2],obj_feat.shape[3])
                        match_map,_ = net.match(obj_feat,feat)
                        # sub_ch_mask, _,_,_ = cv_gen_gaussian(
                        #     np_box_resize(box,img_size,match_map.shape[2:],'polyxy'),
                        #     None,match_map.shape[2:],affin=False)
                        sub_ch_mask = cv_gen_gaussian_by_poly(
                            np_box_resize(box,img_size,match_map.shape[2:],'polyxy'),
                            match_map.shape[2:]
                        )
                        # sub_ch_mask /= np.max(sub_ch_mask)
                        loss_dict['sub_ch_loss'] += criteria(match_map[:,0],torch.from_numpy(np.expand_dims(sub_ch_mask,0)).to(match_map.device))
                        match_map_np = match_map.detach().to('cpu').numpy()
                        cnt+=1
                    except:
                        continue
                if(loss_dict['sub_ch_loss']==0.0):
                    continue
                if(cnt>0):
                    loss_dict['sub_ch_loss']/=cnt
                    try:
                        img = x.numpy().astype(np.uint8)
                        sub_img = sub_img.astype(np.uint8)
                        img[0:sub_img.shape[0],0:sub_img.shape[1],:]=sub_img
                        img = cv_draw_poly(img,np_box_resize(box,x.shape[:-1],img.shape[:-1],'polyxy'))

                        img_msk = cv_mask_image(img,match_map_np[0,0])
                        if(logger):
                            logger.add_image('Tracking result', img_msk, batch*batch_size+i,dataformats='HWC')
                        if(work_dir):
                            save_image(os.path.join(work_dir,'images','tracking','pred','pred_tracking_'+sample['name'][i]),img_msk)
                            
                        img_msk = cv_mask_image(img,sub_ch_mask)
                        if(logger):
                            logger.add_image('Tracking GT', img_msk, batch*batch_size+i,dataformats='HWC')
                        if(work_dir):
                            save_image(os.path.join(work_dir,'images','tracking','gt','gt_tracking_'+sample['name'][i]),img_msk)
                    except:
                        None

                    loss = 0.0
                    loss_log = 'step {},'.format(batch*batch_size+i)
                    for o in loss_dict:
                        loss+=loss_dict[o]
                        loss_log+=' {} {}'.format(o,loss_dict[o].item())
                    if(logger):
                        for o in loss_dict:
                            logger.add_scalar('Loss/'+o, loss_dict[o].item(), batch*batch_size+i)
                        logger.add_scalar('Loss/total loss'.format(batch), loss.item(), batch*batch_size+i)
                        logger.flush()
                    # print("step {}, sub_ch_loss {}, chloss {}, afloss {}.".format(batch*batch_size+i,sub_ch_loss.item(),ch_loss.item(),af_loss.item()))
                    print(loss_log)
                    if(torch.isnan(loss)):
                        return -1
                    loss.backward()
                    opt.step()
                del pred
                del feat
                del obj_map
                del obj_feat
                del match_map
                del x_nor
                del sub_img_nor
                del sample
                del loss_dict
                del loss
            # except Exception as e:
            #     print("Faild in training image {}, step {}, err: {}".format(sample['name'][i],batch*batch_size+i,e))
    return 0
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
    work_dir = DEF_WORK_DIR
    work_dir = os.path.join(work_dir,'log','siam_img')
    log_step_size = args.logstp

    # For Debug config
    lod_dir = "/home/yomcoding/Pytorch/MyResearch/saved_model/siam_craft_img"
    sav_dir = "/home/yomcoding/Pytorch/MyResearch/saved_model/siam_craft_img2"
    # isdebug = True
    # use_net = 'craft_mob'
    # use_dataset = 'minetto'
    # log_step_size = 1
    # use_cuda = False
    # num_workers=0
    # lr_decay_step_size = None

    dev = 'cuda' if(use_cuda)else 'cpu'
    basenet = torch.load("/home/yomcoding/Pytorch/MyResearch/pre_train/craft_mlt_25k.pkl").float()
    net = SiameseCRAFT(base_net=basenet,feature_chs=32).float().to(dev)
    if(lod_dir):
        net.load_state_dict(torch.load(lod_dir+'.pth'))
    # net = net.float().to(dev)
    # net = torch.load(lod_dir).float().to(dev)
    # if(os.path.exists(args.opt)):
    #     opt = torch.load(args.opt)
    # elif(args.opt.lower()=='adam'):
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=tcfg['OPT_DEC'])
    # elif(args.opt.lower() in ['adag','adagrad']):
    #     opt = optim.Adagrad(net.parameters(), lr=lr, weight_decay=tcfg['OPT_DEC'])
    # else:
    #     opt = optim.SGD(net.parameters(), lr=lr, momentum=tcfg['MMT'], weight_decay=tcfg['OPT_DEC'])
    # opt = torch.load('/home/yomcoding/Pytorch/MyResearch/saved_model/siam_craft_opt2.pkl')
    # opt.add_param_group(net.parameters())
    train_dataset = Total(
        os.path.join(DEF_TTT_DIR,'images','train'),
        os.path.join(DEF_TTT_DIR,'gt_pixel','train'),
        os.path.join(DEF_TTT_DIR,'gt_txt','train'),
        out_box_format='polyxy',
        )
    # train_dataset = ICDAR13(
    #     os.path.join(DEF_IC13_DIR,'images','train'),
    #     os.path.join(DEF_IC13_DIR,'gt_txt','train'),
    #     out_box_format='polyxy',
    #     max_image_size=(720,1280),
    #     )
    num_workers = 0
    batch = 2

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
    logger = None
    logger = SummaryWriter(work_dir)
    # img_dir = os.path.join(work_dir,train_dataset.__class__.__name__)
    img_dir = None
    # try:
    ret = train_siam(dataloader,net,opt,loss,len(train_dataset),logger,img_dir)
    # except Exception as e:
    #     print(e)
    
    if(ret==0 and sav_dir):
        print("Saving model...")
        torch.save(net.state_dict(),sav_dir+'.pth')
        print("Saving optimizer...")
        torch.save(opt,sav_dir+'_opt.pkl')

    time_usage = datetime.now()
    print("End at: {}.\n".format(time_usage.strftime("%Y%m%d-%H%M%S")))
    time_usage = time_usage - time_start
    print("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    pass
