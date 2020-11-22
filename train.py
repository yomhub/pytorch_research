import os
import sys
import platform
import torch
import argparse
from datetime import datetime
# =================Torch=======================
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# =================Local=======================
from lib.model.craft import CRAFT,CRAFT_MOB,CRAFT_LSTM,CRAFT_MOTION,CRAFT_VGG_LSTM
from lib.loss.mseloss import MSE_OHEM_Loss
from lib.dataloader.total import Total
from lib.dataloader.icdar import *
from lib.dataloader.icdar_video import ICDARV
from lib.dataloader.minetto import Minetto
from lib.dataloader.base import BaseDataset
from lib.dataloader.synthtext import SynthText
from lib.utils.img_hlp import RandomScale
from lib.fr_craft import CRAFTTrainer
from lib.config.train_default import cfg as tcfg
from dirs import *

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
    parser.add_argument('--epoch', type=int, help='Epoch size.',default=tcfg['EPOCH'])

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
    epoch = args.epoch

    # For Debug config
    # lod_dir = "/home/yomcoding/Pytorch/MyResearch/saved_model/craft_mob_nopd_syn.pkl"
    # teacher_pkl_dir = "/home/yomcoding/Pytorch/MyResearch/pre_train/craft_mlt_25k.pkl"
    # isdebug = True
    # use_net = 'craft_mob'
    # use_dataset = 'sync'
    # log_step_size = 1
    # use_cuda = False
    # num_workers=0
    # lr_decay_step_size = None
    if(lod_dir and os.path.exists(lod_dir)):
        print("Loading at {}".format(lod_dir))
        net = torch.load(lod_dir)
    elif(use_net=='craft'):
        net = CRAFT()
    elif(use_net=='craft_mob'):
        net = CRAFT_MOB(pretrained=True,padding=False)
    elif(use_net=='craft_lstm'):
        net = CRAFT_LSTM()
    elif(use_net=='craft_motion'):
        net = CRAFT_MOTION()
    elif(use_net=='craft_vgg_lstm'):
        net = CRAFT_VGG_LSTM()
    
    net = net.float().to("cuda:0" if(use_cuda)else "cpu")
    
    if(os.path.exists(args.opt)):
        opt = torch.load(args.opt)
    elif(args.opt.lower()=='adam'):
        opt = optim.Adam(net.parameters(), lr=lr, weight_decay=tcfg['OPT_DEC'])
    elif(args.opt.lower() in ['adag','adagrad']):
        opt = optim.Adagrad(net.parameters(), lr=lr, weight_decay=tcfg['OPT_DEC'])
    else:
        opt = optim.SGD(net.parameters(), lr=lr, momentum=tcfg['MMT'], weight_decay=tcfg['OPT_DEC'])

        
    if(use_dataset=="ttt"):
        train_dataset = Total(
            os.path.join(DEF_TTT_DIR,'images','train'),
            os.path.join(DEF_TTT_DIR,'gt_pixel','train'),
            os.path.join(DEF_TTT_DIR,'gt_txt','train'),
            image_size=(640, 640),)
        train_on_real = True
        x_input_function = train_dataset.x_input_function
        y_input_function = None
    elif(use_dataset=="ic15"):
        train_dataset = ICDAR15(
            os.path.join(DEF_IC15_DIR,'images','train'),
            os.path.join(DEF_IC15_DIR,'gt_txt','train'),
            image_size=(640, 640),)
        train_on_real = True
        x_input_function = train_dataset.x_input_function
        y_input_function = None
    elif(use_dataset=='sync'):
        train_dataset = SynthText(DEF_SYN_DIR, 
            image_size=(640, 640),
            )
        train_on_real = False
        x_input_function=train_dataset.x_input_function
        y_input_function=train_dataset.y_input_function
    elif(use_dataset=='icv15'):
        train_dataset = ICDARV(os.path.join(DEF_ICV15_DIR,'train'))
        train_on_real = True
        x_input_function = None
        y_input_function = None
        num_workers = 0
        batch = 1
    elif(use_dataset in ['minetto','mine']):
        train_dataset = Minetto(DEF_MINE_DIR)
        train_on_real = True
        x_input_function = None
        y_input_function = None
        num_workers = 0
        batch = 1
    else:
        # Load ALL jpg/png/bmp image from dir list
        train_dataset = BaseDataset((
            os.path.join(DEF_IC15_DIR,'images','train'),
            os.path.join(DEF_IC19_DIR,'train'),
            os.path.join(DEF_TTT_DIR,'images','train'),
            os.path.join(DEF_MSRA_DIR,'train'),
            DEF_SVT_DIR,
            # __DEF_SYN_DIR,
            ),
            image_size=(640, 640),
            img_only=True)
        train_on_real = True
        x_input_function = train_dataset.x_input_function
        y_input_function = None

    dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if(num_workers>0)else False,
        collate_fn=train_dataset.default_collate_fn,
        )

    loss = MSE_OHEM_Loss(positive_mult = 3,positive_th = 0)
    trainer = CRAFTTrainer
    
    trainer = trainer(
        work_dir = work_dir,
        task_name=args.name if(args.name!=None)else net.__class__.__name__,
        isdebug = isdebug, use_cuda = use_cuda,
        net = net, loss = loss, opt = opt,
        log_step_size = log_step_size,
        save_step_size = args.savestep,
        lr_decay_step_size = lr_decay_step_size, lr_decay_multi = tcfg['LR_DEC_RT'],
        custom_x_input_function=x_input_function,
        custom_y_input_function=y_input_function,
        train_on_real = train_on_real,
        auto_decay=False,
        )
    if(teacher_pkl_dir):
        trainer.set_teacher(teacher_pkl_dir)

    summarize = "Start when {}.\n".format(time_start.strftime("%Y%m%d-%H%M%S")) +\
        "Working DIR: {}\n".format(work_dir)+\
        "Running with: \n"+\
        "\t Step size: {},\n\t Batch size: {}.\n".format(max_step,batch)+\
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
        "\t Save step: {}.\n".format(args.savestep)+\
        "\t Is debug: {}.\n".format('Yes' if(isdebug)else 'No')+\
        ""
    print(summarize)
    trainer.log_info(summarize)
    
    for i in range(epoch):
        trainer.loader_train(dataloader,int(len(train_dataset)/dataloader.batch_size) if(max_step<0)else max_step)
        if(args.save):
            print("Saving model...")
            trainer.save(args.save)
            print("Saving optimizer...")
            trainer.save_opt(args.save+'_opt.pkl')

    time_usage = datetime.now()
    print("End at: {}.\n".format(time_usage.strftime("%Y%m%d-%H%M%S")))
    time_usage = time_usage - time_start
    print("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    pass
