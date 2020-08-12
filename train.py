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
from lib.model.craft import CRAFT
from lib.model.mobilenet_v2 import CRAFT_MOB
from lib.loss.mseloss import MSE_OHEM_Loss, MSELoss
from lib.dataloader.total import Total
from lib.dataloader.icdar_video import ICDARV
import lib.dataloader.synthtext as syn80k
from lib.utils.img_hlp import RandomScale
from lib.trainer_craft import CRAFTTrainer
from lib.config.train_default import cfg as tcfg


__DEF_LOCAL_DIR = os.path.dirname(os.path.realpath(__file__))
__DEF_DATA_DIR = os.path.join(__DEF_LOCAL_DIR, 'dataset')
__DEF_CTW_DIR = os.path.join(__DEF_DATA_DIR, 'ctw')
__DEF_SVT_DIR = os.path.join(__DEF_DATA_DIR, 'svt')
__DEF_TTT_DIR = os.path.join(__DEF_DATA_DIR, 'totaltext')
__DEF_ICV_DIR = os.path.join(__DEF_DATA_DIR, 'TextVideo')
if(platform.system().lower()[:7]=='windows'):__DEF_SYN_DIR = "D:\\development\\SynthText"
elif(os.path.exists("/BACKUP/yom_backup/SynthText")):__DEF_SYN_DIR = "/BACKUP/yom_backup/SynthText"
else:__DEF_SYN_DIR = os.path.join(__DEF_DATA_DIR, 'SynthText')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config trainer')

    parser.add_argument('--opt', help='Choose optimizer.',default=tcfg['OPT'])
    parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
    parser.add_argument('--save', type=str, help='Set --save file_dir if want to save network.')
    parser.add_argument('--load', type=str, help='Set --load file_dir if want to load network.')
    parser.add_argument('--net', help='Choose noework (craft).', default="craft")
    parser.add_argument('--name', help='Name of task.')
    parser.add_argument('--dataset', help='Choose dataset: ctw/svt/ttt.', default=tcfg['DATASET'])
    parser.add_argument('--datax', type=int, help='Dataset output width.',default=tcfg['IMG_SIZE'][0])
    parser.add_argument('--datay', type=int, help='Dataset output height.',default=tcfg['IMG_SIZE'][1])
    parser.add_argument('--step', type=int, help='Step size.',default=tcfg['STEP'])
    parser.add_argument('--batch', type=int, help='Batch size.',default=tcfg['BATCH'])
    parser.add_argument('--logstp', type=int, help='Log step size.',default=tcfg['LOGSTP'])
    parser.add_argument('--gpu', type=int, help='Set --gpu -1 to disable gpu.',default=0)
    parser.add_argument('--savestep', type=int, help='Batch size.',default=20)
    parser.add_argument('--learnrate', type=float, help='Learning rate.',default=tcfg['LR'])

    args = parser.parse_args()
    time_start = datetime.now()
    isdebug = args.debug
    # isdebug = True
    lr = args.learnrate
    max_step = args.step if(not isdebug)else 1
    use_cuda = True if(args.gpu>=0 and torch.cuda.is_available())else False
    lr_decay_step_size = tcfg['LR_DEC_STP']
    num_workers=4 if(platform.system().lower()[:7]!='windows')else 0
    num_workers=0
    work_dir = "/BACKUP/yom_backup" if(platform.system().lower()[:7]!='windows' and os.path.exists("/BACKUP/yom_backup"))else __DEF_LOCAL_DIR
    # lr_decay_step_size = None

    net = CRAFT_MOB()
    if(args.opt.lower()=='adam'):
        opt = optim.Adam(net.parameters(), lr=lr, weight_decay=tcfg['OPT_DEC'])
    else:
        opt = optim.SGD(net.parameters(), lr=lr, momentum=tcfg['MMT'], weight_decay=tcfg['OPT_DEC'])
    train_dataset = syn80k.SynthText(__DEF_SYN_DIR, image_size=(3,640, 640), 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]
    ))
    dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if(num_workers>0)else False)
    loss = MSELoss()
    trainer = CRAFTTrainer
    
    trainer = trainer(
        work_dir = work_dir,
        task_name=args.name if(args.name!=None)else net.__class__.__name__,
        isdebug = isdebug, use_cuda = use_cuda,
        net = net, loss = loss, opt = opt,
        log_step_size = tcfg['LOGSTP'],
        save_step_size = tcfg['SAVESTP'],
        lr_decay_step_size = lr_decay_step_size, lr_decay_multi = tcfg['LR_DEC_RT'],
        custom_x_input_function=syn80k.x_input_function,
        custom_y_input_function=syn80k.y_input_function,
        )

    summarize = "Start when {}.\n".format(time_start.strftime("%Y%m%d-%H%M%S")) +\
        "Working DIR: {}\n".format(work_dir)+\
        "Running with: \n"+\
        "\t Step size: {},\n\t Batch size: {}.\n".format(max_step,args.batch)+\
        "\t Input shape: x={},y={}.\n".format(args.datax,args.datay)+\
        "\t Network: {}.\n".format(net.__class__.__name__)+\
        "\t Optimizer: {}.\n".format(opt.__class__.__name__)+\
        "\t Init learning rate: {}.\n".format(lr)+\
        "\t Taks name: {}.\n".format(args.name if(args.name!=None)else net.__class__.__name__)+\
        "\t Use GPU: {}.\n".format('Yes' if(use_cuda>=0)else 'No')+\
        "\t Save network: {}.\n".format(args.save if(args.save)else 'No')+\
        "\t Load network: {}.\n".format(args.load if(args.load)else 'No')+\
        "\t Is debug: {}.\n".format('Yes' if(isdebug)else 'No')+\
        ""
    print(summarize)
    trainer.log_info(summarize)

    if(args.load):
        print("Loading model...")
        trainer.load(args.load)
    trainer.loader_train(dataloader,int(len(train_dataset)/args.batch) if(max_step<0)else max_step)
    if(args.save):
        print("Saving model...")
        trainer.save(args.save)
    time_usage = datetime.now()
    print("End at: {}.\n".format(time_usage.strftime("%Y%m%d-%H%M%S")))
    time_usage = time_usage - time_start
    print("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    pass
