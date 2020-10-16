import os, sys
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# =================
from lib.utils.log_hlp import str2time
from lib.utils.img_hlp import *

class Tester():
    def __init__(self,  
        work_dir = os.getcwd(),
        log_floder = 'eval', task_name = None,
        isdebug = False, use_cuda = True,
        net = None, loss = None,
        log_step_size = None,
        custom_x_input_function=None,custom_y_input_function=None,
    ):
        self._isdebug = bool(isdebug)
        self._logs_path = os.path.join(work_dir,log_floder)
        if(task_name!=None): self._logs_path = os.path.join(self._logs_path,task_name)
        if(self._isdebug): self._logs_path = os.path.join(self._logs_path,'debug')
        self._logs_path = os.path.join(self._logs_path,datetime.now().strftime("%Y%m%d-%H%M%S")) 
        self._task_name = task_name

        if(not(self._isdebug) and not(os.path.exists(self._logs_path))):
            os.makedirs(self._logs_path,exist_ok=True)
        self._f_train_loger = open(os.path.join(self._logs_path,'eval.txt'),'w',encoding='utf8') if(not self._isdebug)else sys.stdout

        self._file_writer = SummaryWriter(os.path.join(self._logs_path,'Summary')) if(not self._isdebug)else None
        self._use_cuda = bool(use_cuda) and torch.cuda.is_available()
        self._device = torch.device("cuda:0" if self._use_cuda else "cpu")

        self._current_step = 0
        self._batch_size = 0
        self._data_count = 0
        self._task_name = task_name
        self._log_step_size = log_step_size
        self._custom_x_input_function=custom_x_input_function
        self._custom_y_input_function=custom_y_input_function
        if(net==None):
            self._net = None
        elif(isinstance(net,str)):
            try:
                self._net = torch.load(net)
                self._net.float().to(self._device)
                self._net.eval()
            except:
                self._net = None
        else:
            self._net = net.float().to(self._device)
            self._net.eval()
        self._loss = loss.to(self._device) if(loss!=None)else None

    def set_tester(self,net=None,loss=None,cmd=None,custom_x_input_function=None,custom_y_input_function=None):
        if(net==None):
            self._net = None
        elif(isinstance(net,str)):
            try:
                self._net = torch.load(net)
                self._net.float().to(self._device)
                self._net.eval()
            except:
                self._net = None
        else:
            self._net = net.float().to(self._device)
            self._net.eval()
        if(loss!=None):self._loss = loss.to(self._device)
        if(custom_x_input_function!=None):self._custom_x_input_function = custom_x_input_function
        if(custom_y_input_function!=None):self._custom_y_input_function = custom_y_input_function

    def loader_test(self,loader,test_size,info=None):
        if(self._net==None):
            print("Tester err: net is None")
            return -1
        if(self._custom_x_input_function==None or self._custom_x_input_function==None):
            print("Tester err: custom_input_function is None")
            return -1
        
        batch_size = loader.batch_size

        self._f_train_loger.write("Step {}, batch size = {}, device = {}.\n".format(self._current_step,batch_size,self._device))

        # with torch.autograd.profiler.profile() as prof:
        with tqdm(total=min(test_size,100)) as pbar:
            for j,sample in enumerate(loader):
                if(j>=test_size):break
                x=self._custom_x_input_function(sample,self._device)
                y=self._custom_y_input_function(sample,self._device) if(self._custom_y_input_function!=None)else None
                x = x.permute(0,2,3,1)
                # x = torch.nn.functional.pad(x,(0,0,0,0,0,640-x.shape[1]),"constant", 0)
                x = torch_img_normalize(x).permute(0,3,1,2)
                # noisy = torch.rand((x.shape[0],x.shape[1],640-x.shape[2],x.shape[3]),device=x.device)
                # x = torch.cat((x,noisy),dim=2)
                with torch.no_grad():
                    pred = self._net(x)
                    loss = self._loss(pred,y) if(self._loss!=None and y!=None)else None
                cryt = self._criterion(x,pred,sample)
                self._step_callback(sample,x,y,pred,loss.item()if(loss!=None)else None,cryt,self._current_step,batch_size)

                if(not(self._isdebug) and self._log_step_size!=None and self._log_step_size>0 and self._current_step%self._log_step_size==0):
                    self._logger(sample,x,y,pred,loss.item() if(loss!=None)else None,cryt,self._current_step,batch_size)
                    if(self._file_writer!=None):self._file_writer.flush()

                if(loss!=None):
                    self._f_train_loger.write("Avg loss:{}.\n".format(loss.item()))
                self._f_train_loger.flush()

                # del 
                del sample
                del x
                del y
                self._current_step += 1

                if(test_size<=100 or j%int(test_size/100)==0):
                    pbar.update()

        # self._f_train_loger.write(str(prof))
        self._f_train_loger.flush()
        return 0

    def _logger(self,sample,x,y,pred,loss,cryt,step,batch_size):
        return None
    def _step_callback(self,sample,x,y,pred,loss,cryt,step,batch_size):
        return None
    def _criterion(self,x,pred,sample):
        return None
        
    def get_net_size(self):
        if(self._net==None):return 0
        return sum(param.numel() for param in self._net.parameters())

    def log_info(self,info):
        if(self._f_train_loger!=None and info!=None):
            self._f_train_loger.write(info)
            self._f_train_loger.flush()
