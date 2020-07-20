import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.log_hlp import str2time

__DEF_PROJ_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class Trainer():
    def __init__(self,  
        work_dir = __DEF_PROJ_PATH, model_floder = 'saved_model',
        log_floder = 'log', task_name = None,
        isdebug = False, use_cuda = True,
        net = None, loss = None, opt = None,
    ):
        self._isdebug = bool(isdebug)
        self._logs_path = os.path.join(work_dir,log_floder)
        if(task_name!=None): self._logs_path = os.path.join(self._logs_path,task_name)
        if(self._isdebug): self._logs_path = os.path.join(self._logs_path,'debug')
        self._logs_path = os.path.join(self._logs_path,datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._f_train_loger = open(od.path.join(self._logs_path,'train.txt'),'w',encoding='utf8')
        self._model_path = os.path.join(work_dir,model_floder)
        if(not(os.path.exists(self._logs_path))):
            os.makedirs(self._logs_path,exist_ok=True)

        self._file_writer = SummaryWriter(self._logs_path) if(self._isdebug)else None

        self.grad_dict = {}
        self.loss_dict = {}
        self._current_step = 0
        self._batch_size = 0
        self._data_count = 0
        self._model = None
        self._loss = None
        self._opt = None
        self._eva_step = 0
        self._task_name = task_name

        self._net = net
        self._opt = opt
        self._loss = loss

        self._use_cuda = bool(use_cuda) and torch.cuda.is_available()
        self._device = torch.device("cuda:0" if self._use_cuda else "cpu")

    def set_trainer(self,net=None,opt=None,loss=None,cmd=None):
        if(net!=None):self._net = net
        if(opt!=None):self._opt = opt
        if(loss!=None):self._loss = loss
    
    def train(self,xs,ys,info=None,custom_x_input_function=None,custom_y_input_function=None):
        if(self._net==None or self._opt==None or self._loss==None):return -1

        self._f_train_loger.write(info+'\n')
        if(isinstance(xs,list) or isinstance(xs,tuple)):batch_size=len(xs)
        else:
            batch_size= xs.shape[0] if(isinstance(xs,torch.Tensor))else 'Unknow'

        if(isinstance(ys,list or isinstance(xs,tuple)) and len(ys)!=batch_size):return -1
        elif(isinstance(ys,torch.Tensor)):
            if(ys.shape[0]!=batch_size):return -1
            ys = torch.split(ys,1)

        self._f_train_loger.write("Step {}, batch size = {}, device = {}.\n".format(self._current_step,batch_size,self._device))

        self._net.to(self._device)
        c_loss = 0.0
        with torch.autograd.profiler.profile() as prof:
            for i in range(len(xs)):
                x=xs[i]
                y=ys[i]
                if(custom_x_input_function!=None):
                    x=custom_x_input_function(x,self._device)
                else:
                    if(self._use_cuda and isinstance(x,torch.Tensor)): x = x.to(self._device)
                if(custom_y_input_function!=None):
                    y=custom_y_input_function(y,self._device)
                else:
                    if(self._use_cuda and isinstance(y,torch.Tensor)): y = y.to(self._device)

                self._opt.zero_grad()
                pred = self._net(x)
                loss = self._loss(pred,y)
                loss.backward()
                self._opt.step()
                c_loss += loss.item()

        c_loss /= float(len(xs))
        self._current_step += 1
        self._f_train_loger.write(prof)
        self._f_train_loger.flush()
        return 0

    def _logger(self,):


    def test(self,datas):
        if(self.net==None):return
        
    
