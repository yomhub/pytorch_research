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
from lib.model.craft import CRAFT,CRAFT_MOB,CRAFT_LSTM,CRAFT_MOTION
from lib.loss.mseloss import MSE_OHEM_Loss
from lib.dataloader.total import Total
from lib.dataloader.icdar import ICDAR
from lib.dataloader.icdar_video import ICDARV
from lib.dataloader.minetto import Minetto
from lib.dataloader.base import BaseDataset
import lib.dataloader.synthtext as syn80k
from lib.utils.img_hlp import RandomScale
from lib.fr_craft import CRAFTTrainer
from lib.config.train_default import cfg as tcfg


__DEF_LOCAL_DIR = os.path.dirname(os.path.realpath(__file__))
__DEF_DATA_DIR = os.path.join(__DEF_LOCAL_DIR, 'dataset')
__DEF_CTW_DIR = os.path.join(__DEF_DATA_DIR, 'ctw')
__DEF_SVT_DIR = os.path.join(__DEF_DATA_DIR, 'svt', 'img')
__DEF_TTT_DIR = os.path.join(__DEF_DATA_DIR, 'totaltext')
__DEF_IC15_DIR = os.path.join(__DEF_DATA_DIR, 'ICDAR2015')
__DEF_IC19_DIR = os.path.join(__DEF_DATA_DIR, 'ICDAR2019')
__DEF_ICV15_DIR = os.path.join(__DEF_DATA_DIR, 'ICDAR2015_video')
__DEF_MINE_DIR = os.path.join(__DEF_DATA_DIR, 'minetto')

if(platform.system().lower()[:7]=='windows'):__DEF_SYN_DIR = "D:\\development\\SynthText"
elif(os.path.exists("/BACKUP/yom_backup/SynthText")):__DEF_SYN_DIR = "/BACKUP/yom_backup/SynthText"
else:__DEF_SYN_DIR = os.path.join(__DEF_DATA_DIR, 'SynthText')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config trainer')

    parser.add_argument('--opt', help='Choose optimizer.',default=tcfg['OPT'])
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

    # For Debug config
    # lod_dir = "/home/yomcoding/Pytorch/MyResearch/saved_model/craft_lstm.pkl"
    # teacher_pkl_dir = "/home/yomcoding/Pytorch/MyResearch/pre_train/craft_mlt_25k.pkl"
    # isdebug = True
    # use_net = 'craft_mob'
    # use_dataset = 'all'
    # use_cuda = False
    # num_workers=0
    # lr_decay_step_size = None

    if(use_cuda):
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.distributed.init_process_group()

    if(use_net=='craft'):
        net = CRAFT()
    elif(use_net=='craft_mob'):
        net = CRAFT_MOB(pretrained=True)
    elif(use_net=='craft_lstm'):
        net = CRAFT_LSTM()
    elif(use_net=='craft_motion'):
        net = CRAFT_MOTION()

    net = net.float().to("cuda:0" if(use_cuda)else "cpu")
    
    if(args.opt.lower()=='adam'):
        opt = optim.Adam(net.parameters(), lr=lr, weight_decay=tcfg['OPT_DEC'])
    elif(args.opt.lower() in ['adag','adagrad']):
        opt = optim.Adagrad(net.parameters(), lr=lr, weight_decay=tcfg['OPT_DEC'])
    else:
        opt = optim.SGD(net.parameters(), lr=lr, momentum=tcfg['MMT'], weight_decay=tcfg['OPT_DEC'])

        
    if(use_dataset=="ttt"):
        train_dataset = Total(
            os.path.join(__DEF_TTT_DIR,'Images','Train'),
            os.path.join(__DEF_TTT_DIR,'gt_pixel','Train'),
            os.path.join(__DEF_TTT_DIR,'gt_txt','Train'),
            image_size=(3,640, 640),)
        train_on_real = True
        x_input_function = train_dataset.x_input_function
        y_input_function = None
    elif(use_dataset=="ic15"):
        train_dataset = ICDAR(
            os.path.join(__DEF_IC15_DIR,'images','train'),
            os.path.join(__DEF_IC15_DIR,'gt_txt','train'),
            image_size=(3,640, 640),)
        train_on_real = True
        x_input_function = train_dataset.x_input_function
        y_input_function = None
    elif(use_dataset=='sync'):
        train_dataset = syn80k.SynthText(__DEF_SYN_DIR, image_size=(3,640, 640), 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ]
        ))
        train_on_real = False
        x_input_function=syn80k.x_input_function
        y_input_function=syn80k.y_input_function
    elif(use_dataset=='icv15'):
        train_dataset = ICDARV(os.path.join(__DEF_ICV15_DIR,'train'))
        train_on_real = True
        x_input_function = None
        y_input_function = None
        num_workers = 0
        batch = 1
    elif(use_dataset in ['minetto','mine']):
        train_dataset = Minetto(__DEF_MINE_DIR)
        train_on_real = True
        x_input_function = None
        y_input_function = None
        num_workers = 0
        batch = 1
    else:
        train_dataset = BaseDataset((
            os.path.join(__DEF_IC15_DIR,'images','train'),
            os.path.join(__DEF_IC19_DIR,'Train'),
            os.path.join(__DEF_TTT_DIR,'Images','Train'),
            __DEF_SVT_DIR,
            # __DEF_SYN_DIR,
            ),
            image_size=(3,640, 640),
            img_only=True)
        train_on_real = True
        x_input_function = train_dataset.x_input_function
        y_input_function = None

    dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if(num_workers>0)else False,
        collate_fn=train_dataset.default_collate_fn,
        )

    loss = MSE_OHEM_Loss()
    trainer = CRAFTTrainer
    
    trainer = trainer(
        work_dir = work_dir,
        task_name=args.name if(args.name!=None)else net.__class__.__name__,
        isdebug = isdebug, use_cuda = use_cuda,
        net = net, loss = loss, opt = opt,
        log_step_size = args.logstp,
        save_step_size = args.savestep,
        lr_decay_step_size = lr_decay_step_size, lr_decay_multi = tcfg['LR_DEC_RT'],
        custom_x_input_function=x_input_function,
        custom_y_input_function=y_input_function,
        train_on_real = train_on_real,
        )
    if(teacher_pkl_dir):
        trainer.set_teacher(teacher_pkl_dir)

    summarize = "Start when {}.\n".format(time_start.strftime("%Y%m%d-%H%M%S")) +\
        "Working DIR: {}\n".format(work_dir)+\
        "Running with: \n"+\
        "\t Step size: {},\n\t Batch size: {}.\n".format(max_step,batch)+\
        "\t Input shape: x={},y={}.\n".format(args.datax,args.datay)+\
        "\t Network: {}.\n".format(net.__class__.__name__)+\
        "\t Optimizer: {}.\n".format(opt.__class__.__name__)+\
        "\t Dataset: {}.\n".format(train_dataset.__class__.__name__)+\
        "\t Init learning rate: {}.\n".format(lr)+\
        "\t Learning rate decay: {}.\n".format(lr_decay_step_size if(lr_decay_step_size>0)else "Disabled")+\
        "\t Taks name: {}.\n".format(args.name if(args.name!=None)else net.__class__.__name__)+\
        "\t Teacher: {}.\n".format(teacher_pkl_dir)+\
        "\t Use GPU: {}.\n".format('Yes' if(use_cuda>=0)else 'No')+\
        "\t Load network: {}.\n".format(lod_dir if(lod_dir)else 'No')+\
        "\t Save network: {}.\n".format(args.save if(args.save)else 'No')+\
        "\t Is debug: {}.\n".format('Yes' if(isdebug)else 'No')+\
        ""
    print(summarize)
    trainer.log_info(summarize)
    
    if(lod_dir):
        print("Loading model at {}.".format(lod_dir))
        trainer.load(lod_dir)

    trainer.loader_train(dataloader,int(len(train_dataset)/dataloader.batch_size) if(max_step<0)else max_step)
    if(args.save):
        print("Saving model...")
        trainer.save(args.save)
    time_usage = datetime.now()
    print("End at: {}.\n".format(time_usage.strftime("%Y%m%d-%H%M%S")))
    time_usage = time_usage - time_start
    print("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    pass
