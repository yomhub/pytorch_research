import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.log_hlp import str2time

__DEF_PROJ_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class Trainer():
    def __init__(self,  
        work_dir = __DEF_PROJ_PATH, model_floder = 'save_model',
        log_floder = 'log', task_name = None,
        isdebug = False, use_cuda = True,
        net = None, loss = None, opt = None,
    ):
        self.isdebug = bool(isdebug)
        self.logs_path = os.path.join(work_dir,log_floder)
        if(task_name!=None): self.logs_path = os.path.join(self.logs_path,task_name)
        if(self.isdebug): self.logs_path = os.path.join(self.logs_path,'debuge')
        self.logs_path = os.path.join(self.logs_path,datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model_path = os.path.join(work_dir,model_floder)

        if(not(os.path.exists(self.logs_path))):
            os.makedirs(self.logs_path,exist_ok=True)

        self.file_writer = SummaryWriter(self.logs_path) if(self.isdebug)else None

        self.grad_dict = {}
        self.loss_dict = {}
        self.current_step = 0
        self.batch = 0
        self.data_count = 0
        self.model = None
        self.loss = None
        self.opt = None
        self.eva_step = 0
        self.task_name = task_name

        self.net = net
        self.opt = opt
        self.loss = loss

        self.use_cuda = bool(use_cuda)

    def set_trainer(self,net=None,opt=None,loss=None):
        if(net!=None):self.net = net
        if(opt!=None):self.opt = opt
        if(loss!=None):self.loss = loss

    def test(self,datas):
        if(self.net==None):return
        
    