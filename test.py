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
from lib.loss.mseloss import MSE_OHEM_Loss
from lib.dataloader.total import Total
from lib.dataloader.icdar_video import ICDARV
import lib.dataloader.synthtext as syn80k
from lib.utils.img_hlp import RandomScale
from lib.fr_craft import CRAFTTester
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


from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config tester')

    parser.add_argument('--opt', help='Choose optimizer.',default=tcfg['OPT'])
    parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
    parser.add_argument('--load', type=str, help='Set --load file_dir if want to load network.')
    parser.add_argument('--name', help='Name of task.')
    parser.add_argument('--dataset', help='Choose dataset: ctw/svt/ttt.', default=tcfg['DATASET'])
    parser.add_argument('--datax', type=int, help='Dataset output width.',default=tcfg['IMG_SIZE'][0])
    parser.add_argument('--datay', type=int, help='Dataset output height.',default=tcfg['IMG_SIZE'][1])
    parser.add_argument('--step', type=int, help='Step size.',default=5)
    parser.add_argument('--logstp', type=int, help='Log step size.',default=tcfg['LOGSTP'])
    parser.add_argument('--gpu', type=int, help='Set --gpu -1 to disable gpu.',default=-1)

    args = parser.parse_args()
    load_dir = args.load
    # load_dir = "/home/yomcoding/Pytorch/MyResearch/pre_train/craft_mlt_25k.pkl"
    load_dir = "/BACKUP/yom_backup/saved_model/CRAFT_MOB_Adag/20200819-010649+craft_MOB_normal_adamg.pkl"
    if(load_dir and os.path.exists(load_dir)):
        net = torch.load(load_dir)
    else:
        raise Exception("Can't load network.")
    # net = CRAFT()
    # net.load_state_dict(copyStateDict(torch.load(load_dir, map_location='cpu')))
    time_start = datetime.now()
    isdebug = args.debug
    # isdebug = True

    max_step = args.step if(not isdebug)else 10
    use_cuda = True if(args.gpu>=0 and torch.cuda.is_available())else False

    num_workers=4 if(platform.system().lower()[:7]!='windows')else 0
    # num_workers=0
    work_dir = "/BACKUP/yom_backup" if(platform.system().lower()[:7]!='windows' and os.path.exists("/BACKUP/yom_backup"))else __DEF_LOCAL_DIR
    # lr_decay_step_size = None

    device = torch.device("cuda:0" if use_cuda else "cpu")
    net.eval()
    net.float().to(device)

    train_dataset = syn80k.SynthText(__DEF_SYN_DIR, image_size=(3,640, 640))

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if(num_workers>0)else False)
    loss = MSE_OHEM_Loss()
    tester = CRAFTTester
    
    tester = tester(
        work_dir = work_dir,
        task_name=args.name if(args.name!=None)else net.__class__.__name__,
        isdebug = isdebug, use_cuda = use_cuda,
        net = net, loss = loss,
        log_step_size = 1,
        custom_x_input_function=syn80k.x_input_function,
        custom_y_input_function=syn80k.y_input_function,
        )

    summarize = "Start when {}.\n".format(time_start.strftime("%Y%m%d-%H%M%S")) +\
        "Working DIR: {}\n".format(work_dir)+\
        "Running with: \n"+\
        "\t Input shape: x={},y={}.\n".format(args.datax,args.datay)+\
        "\t Network: {}.\n".format(net.__class__.__name__)+\
        "\t Taks name: {}.\n".format(args.name if(args.name!=None)else net.__class__.__name__)+\
        "\t Use GPU: {}.\n".format('Yes' if(use_cuda>=0)else 'No')+\
        "\t Load network: {}.\n".format(args.load if(args.load)else 'No')+\
        "\t Is debug: {}.\n".format('Yes' if(isdebug)else 'No')+\
        ""
    print(summarize)
    tester.log_info(summarize)

    tester.loader_test(dataloader,int(len(train_dataset)/args.batch) if(max_step<0)else max_step)

    time_usage = datetime.now()
    print("End at: {}.\n".format(time_usage.strftime("%Y%m%d-%H%M%S")))
    time_usage = time_usage - time_start
    print("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    pass
