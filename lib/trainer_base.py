import os
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# =================
from lib.utils.log_hlp import str2time

DEF_PROJ_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class Trainer():
    def __init__(self,  
        work_dir = DEF_PROJ_PATH, model_floder = 'saved_model',
        log_floder = 'log', task_name = None,
        isdebug = False, use_cuda = True,
        net = None, loss = None, opt = None,
        log_step_size = None, save_step_size = None, 
        lr_decay_step_size = None, lr_decay_multi = None,
        custom_x_input_function=None,custom_y_input_function=None,
    ):
        self._isdebug = bool(isdebug)
        self._logs_path = os.path.join(work_dir,log_floder)
        if(task_name!=None): self._logs_path = os.path.join(self._logs_path,task_name)
        if(self._isdebug): self._logs_path = os.path.join(self._logs_path,'debug')
        self._logs_path = os.path.join(self._logs_path,datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._model_path = os.path.join(work_dir,model_floder)
        self._task_name = task_name

        if(not(os.path.exists(self._logs_path))):
            os.makedirs(self._logs_path,exist_ok=True)
        self._f_train_loger = open(os.path.join(self._logs_path,'train.txt'),'w',encoding='utf8')

        self._file_writer = SummaryWriter(os.path.join(self._logs_path,'Summary')) if(not self._isdebug)else None
        self._use_cuda = bool(use_cuda) and torch.cuda.is_available()
        self._device = torch.device("cuda:0" if self._use_cuda else "cpu")

        self.grad_dict = {}
        self.loss_dict = {}
        self._current_step = 0
        self._batch_size = 0
        self._data_count = 0
        self._eva_step = 0
        self._task_name = task_name
        self._log_step_size = log_step_size
        self._save_step_size = save_step_size
        self._lr_decay_step_size = lr_decay_step_size
        self._lr_decay_multi = lr_decay_multi
        self._custom_x_input_function=custom_x_input_function
        self._custom_y_input_function=custom_y_input_function
        self._net = net.float().to(self._device) if(net!=None)else None
        self._loss = loss.to(self._device) if(loss!=None)else None
        self._opt = opt

    def set_trainer(self,net=None,opt=None,loss=None,cmd=None,custom_x_input_function=None,custom_y_input_function=None):
        if(net!=None):self._net = net.float().to(self._device)
        if(opt!=None):self._opt = opt.to(self._device)
        if(loss!=None):self._loss = loss.to(self._device)
        if(custom_x_input_function!=None):self._custom_x_input_function = custom_x_input_function
        if(custom_y_input_function!=None):self._custom_y_input_function = custom_y_input_function
    
    def train(self,xs,ys,info=None,custom_x_input_function=None,custom_y_input_function=None):
        if(self._net==None or self._opt==None or self._loss==None):return -1
        
        if(info!=None):self._f_train_loger.write(info+'\n')
        if(isinstance(xs,list) or isinstance(xs,tuple)):batch_size=len(xs)
        else:
            batch_size= xs.shape[0] if(isinstance(xs,torch.Tensor))else 'Unknow'

        if(isinstance(ys,list or isinstance(xs,tuple)) and len(ys)!=batch_size):return -1
        elif(isinstance(ys,torch.Tensor)):
            if(ys.shape[0]!=batch_size):return -1
            ys = torch.split(ys,1)

        self._f_train_loger.write("Step {}, batch size = {}, device = {}.\n".format(self._current_step,batch_size,self._device))

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
        if(self._log_step_size!=None and self._current_step%self._log_step_size==0):
            self._logger(x,y,pred,c_loss,self._current_step,batch_size)
        
        self._f_train_loger.write("Avg loss:{}.\n".format(c_loss))
        self._f_train_loger.write(prof)
        self._f_train_loger.flush()
        return 0

    def loader_train(self,loader,train_size,info=None):
        if(self._net==None or self._opt==None or self._loss==None):
            print("Trainer log err: net/opt/loss is None")
            return -1
        if(self._custom_x_input_function==None or self._custom_x_input_function==None):
            print("Trainer log err: custom_input_function is None")
            return -1
        
        batch_size = loader.batch_size

        self._f_train_loger.write("Step {}, batch size = {}, device = {}.\n".format(self._current_step,batch_size,self._device))

        with torch.autograd.profiler.profile() as prof:
            with tqdm(total=min(train_size,100)) as pbar:
                i=0
                for j,sample in enumerate(loader):
                    if(i>=train_size):break
                    x=self._custom_x_input_function(sample,self._device)
                    y=self._custom_y_input_function(sample,self._device)
                    self._opt.zero_grad()
                    pred = self._net(x)
                    loss = self._loss(pred,y)
                    loss.backward()
                    self._opt.step()
                    self._current_step += 1

                    if(self._save_step_size!=None and self._current_step%self._save_step_size==0):
                        self.save()

                    if(self._lr_decay_step_size!=None and self._lr_decay_multi!=None and self._current_step%self._lr_decay_step_size==0):
                        self._f_train_loger.write(
                            "Change learning rate form {} to {}.\n".format(
                                self._opt.param_groups[0]['lr'],self._opt.param_groups[0]['lr']*self._lr_decay_multi))
                        for param_group in self._opt.param_groups:
                            param_group['lr'] *= self._lr_decay_multi
                    
                    if(self._log_step_size!=None and self._current_step%self._log_step_size==0):
                        self._logger(x,y,pred,loss.item(),self._current_step,batch_size)

                    self._f_train_loger.write("Avg loss:{}.\n".format(loss.item()))

                    if(train_size<=100 or i%int(train_size/100)==0):
                        pbar.update()
                    i+=1

        self._f_train_loger.write(str(prof))
        self._f_train_loger.flush()
        return 0

    def _logger(self,x,y,pred,loss,step,batch_size):
        return None

    def get_net_size(self):
        if(self._net==None):return 0
        return sum(param.numel() for param in self._net.parameters())

    def save(self,save_dir=None):
        """
        save_dir:
            None or
            '/xxx/save.pkl' or
            '/xxx/'
        """
        if(self._net==None):return
        now_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = "{}+{}.pkl".format(now_time,self._task_name if(self._task_name!=None)else "save")

        if(save_dir==None): save_dir = os.path.join(self._model_path,file_name)
        elif(len(save_dir.split('.'))==1): save_dir = os.path.join(save_dir,file_name)
        else:
            save_dir = os.path.join(
                os.path.split(save_dir)[0],
                "{}+{}".format(now_time,os.path.split(save_dir)[1]))
            
        if(not os.path.exists(os.path.dirname(save_dir))):
            os.makedirs(os.path.dirname(save_dir),exist_ok=True)
        torch.save(self._net,save_dir)
        return 
        
    def load(self,load_dir=None):
        """
        save_dir:
            None or '/xxx/' for newest file
            '/xxx/***.pkl' for specific file
        """
        if(load_dir==None):load_dir = self._model_path
        if(len(load_dir.split('.'))>1):
            if(os.path.exists(load_dir)):
                self._net=torch.load(load_dir)
                self._net=self._net.float().to(self._device)
                return
            else:
                load_dir = os.path.dirname(load_dir.split('.')[0])
        if(not os.path.exists(load_dir)):
            print("Load failed at {}, folder not exist.".format(load_dir))
            return
        tsk_list = [o for o in os.listdir(load_dir) if o[-4:]=='.pkl']
        last_time = str2time(tsk_list[0].split('+')[0])
        cur_i = 0
        for i,o in enumerate(tsk_list):
            cur_time = str2time(o.split('+')[0])
            if(cur_time>last_time):cur_i=i
        print("Load at {}.".format(os.path.join(load_dir,tsk_list[cur_i])))
        self._net=torch.load(os.path.join(load_dir,tsk_list[cur_i]))
        self._net=self._net.float().to(self._device)
        return 

    def log_info(self,info):
        if(self._f_train_loger!=None):
            self._f_train_loger.write(info)
            self._f_train_loger.flush()