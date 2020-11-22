import os, sys
import heapq
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# =================
from lib.utils.log_hlp import str2time

class Trainer():
    def __init__(self,  
        work_dir = os.getcwd(), model_floder = 'saved_model',
        log_floder = 'log', task_name = None,
        isdebug = False, use_cuda = True,
        net = None, loss = None, opt = None,
        log_step_size:int = None, save_step_size:int = None, 
        lr_decay_step_size:int = None, lr_decay_multi:float = 0.9, auto_decay:bool = False, auto_decay_stp:int = 100,
        custom_x_input_function=None,custom_y_input_function=None,
    ):
        self._isdebug = bool(isdebug)
        self._logs_path = os.path.join(work_dir,log_floder)
        if(task_name!=None): self._logs_path = os.path.join(self._logs_path,task_name)
        if(self._isdebug): self._logs_path = os.path.join(self._logs_path,'debug')
        self._logs_path = os.path.join(self._logs_path,datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._model_path = os.path.join(work_dir,model_floder) if(task_name==None)else os.path.join(work_dir,model_floder,task_name) 
        self._task_name = task_name

        if(not(self._isdebug) and not(os.path.exists(self._logs_path))):
            os.makedirs(self._logs_path,exist_ok=True)
        self._f_train_loger = open(os.path.join(self._logs_path,'train.txt'),'w',encoding='utf8') if(not self._isdebug)else sys.stdout

        self._file_writer = SummaryWriter(os.path.join(self._logs_path,'Summary')) if(not self._isdebug)else None
        self._device = torch.device("cuda:0" if(bool(use_cuda) and torch.cuda.is_available())else "cpu")

        self._current_step = 0
        self._batch_size = 0
        self._data_count = 0
        self._task_name = task_name
        self._log_step_size = log_step_size
        self._save_step_size = save_step_size
        self._lr_decay_step_size = lr_decay_step_size
        self._lr_decay_rate = lr_decay_multi
        self._auto_decay = bool(auto_decay)
        self._auto_decay_stp = int(auto_decay_stp)
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
        if(isinstance(xs,(list,tuple,dict))):batch_size=len(xs)
        elif(isinstance(xs,(torch.Tensor,np.ndarray))):batch_size = xs.shape[0] 
        else: batch_size = 'Unknow'

        if(isinstance(ys,(list,tuple,dict)) and len(ys)!=batch_size):return -1
        elif(sinstance(ys,(torch.Tensor,np.ndarray))):
            if(ys.shape[0]!=batch_size):return -1
            ys = torch.split(ys,1)

        self._f_train_loger.write("Step {}, batch size = {}, device = {}.\n".format(self._current_step,batch_size,self._device))

        c_loss = 0.0
        with torch.autograd.profiler.profile() as prof:
            for i in range(len(xs)):
                sample = (xs[i],ys[i])
                x,y,pred,loss = self._train_act(sample)
                c_loss += loss.item()
                self._step_callback(sample,x,y,pred,loss.item(),self._current_step,batch_size)
        
        c_loss /= float(len(xs))
        self._current_step += 1
        if(self._log_step_size!=None and self._current_step%self._log_step_size==0):
            self._logger(sample,x,y,pred,c_loss,self._current_step,batch_size)
        
        self._f_train_loger.write("Avg loss:{}.\n".format(c_loss))
        self._f_train_loger.write(prof)
        self._f_train_loger.flush()
        return 0

    def loader_train(self,loader,train_size,info=None):
        if(self._net==None or self._opt==None or self._loss==None):
            print("Trainer err: net/opt/loss is None")
            return -1
        if(self._custom_x_input_function==None or self._custom_x_input_function==None):
            print("Warring: custom_input_function is None")
        
        batch_size = loader.batch_size

        self._f_train_loger.write("Step {}, batch size = {}, device = {}.\n".format(self._current_step,batch_size,self._device))

        # with torch.autograd.profiler.profile() as prof:
        with tqdm(total=min(train_size,100)) as pbar:
            i=0
            loss_mean=0.0
            loss_range=0.0
            loss_lst=[]

            for j,sample in enumerate(loader):
                if(i>=train_size):break
                x,y,pred,loss = self._train_act(sample)
                self._step_callback(sample,x,y,pred,loss if(isinstance(loss,list))else loss.item(),self._current_step,batch_size)

                if(not(self._isdebug) and self._log_step_size!=None and self._log_step_size>0 and self._current_step%self._log_step_size==0):
                    try:
                        self._logger(sample,x,y,pred,loss if(isinstance(loss,list))else loss.item(),self._current_step,batch_size)
                        if(self._file_writer!=None):self._file_writer.flush()
                    except Exception as e:
                        print("Log err: {}".format(str(e)))

                if(not isinstance(loss,list) and torch.isnan(loss).item()):
                    self._f_train_loger.write("Nan at:{}.\n".format(self._current_step))
                    return -1

                if(not(self._isdebug) and self._save_step_size!=None and self._save_step_size>0 and self._current_step%self._save_step_size==0):
                    try:
                        self.save()
                    except Exception as e:
                        print("Save err: {}".format(str(e)))

                if(self._lr_decay_step_size!=None and self._lr_decay_step_size>0 and self._lr_decay_rate!=None and (self._current_step+1)%self._lr_decay_step_size==0):
                    try:
                        self.opt_decay(self._lr_decay_rate)
                    except Exception as e:
                        print("Opt_decay err: {}".format(str(e)))

                if(not isinstance(loss,list)):
                    self._f_train_loger.write("Avg loss:{}.\n".format(loss.item()))
                    self._f_train_loger.flush()

                # del 
                del sample
                del x
                del y
                self._current_step += 1

                if(self._auto_decay):
                    loss_lst.append(loss.item())
                    if(len(loss_lst)>=self._auto_decay_stp):
                        loss_lst = np.array(loss_lst)
                        cur_mean = np.mean(loss_lst)
                        cur_range = np.max(loss_lst) - np.min(loss_lst)
                        if(np.abs(loss_mean-cur_mean)/cur_range<0.05):
                            self.opt_decay()
                        loss_mean=cur_mean
                        loss_lst=[]

                if(train_size<=100 or i%int(train_size/100)==0):
                    pbar.update()
                i+=1

        # self._f_train_loger.write(str(prof))
        # self._f_train_loger.flush()
        return 0

    def _train_act(self,sample):
        x=self._custom_x_input_function(sample,self._device)
        y=self._custom_y_input_function(sample,self._device)
        self._opt.zero_grad()
        pred = self._net(x)
        loss = self._loss(pred,y)
        loss.backward()
        self._opt.step()
        return x,y,pred,loss

    def _logger(self,sample,x,y,pred,loss,step,batch_size):
        return None
    def _step_callback(self,sample,x,y,pred,loss,step,batch_size):
        return None

    def get_net_size(self):
        if(self._net==None):return 0
        return sum(param.numel() for param in self._net.parameters())

    def save(self,save_dir=None,max_save=3):
        """
        save_dir:
            'save.pkl' or
            '/xxx/save.pkl' or
            '/xxx/'
        """
        if(self._net==None):return
        now_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = "{}+{}.pkl".format(now_time,self._task_name if(self._task_name!=None)else "save")

        if(save_dir==None): save_dir = os.path.join(self._model_path,file_name)
        elif(os.path.dirname(save_dir)):
            print("Save at {}".format(save_dir))
            torch.save(self._net,save_dir)
            return
        elif(len(save_dir.split('.'))==1): save_dir = os.path.join(save_dir,file_name)
        elif(os.path.dirname(save_dir)==''): 
            save_dir = os.path.join(self._model_path,"{}+{}.pkl".format(now_time,save_dir.split('.')[0]))
        else:
            save_dir = save_dir
            
        if(not os.path.exists(os.path.dirname(save_dir))):
            os.makedirs(os.path.dirname(save_dir),exist_ok=True)

        if(max_save!=None and max_save>0):
            tsk_list = [o for o in os.listdir(os.path.dirname(save_dir)) if o[-4:]=='.pkl']
            if(len(tsk_list)>max_save):
                cur_time = []
                sub_name = tsk_list[0].split('+')[1]
                base_dir = os.path.dirname(save_dir)
                for o in tsk_list:
                    try:
                        heapq.heappush(cur_time,str2time(o.split('+')[0]))
                    except:
                        continue
                for o in cur_time[:1-max_save]:
                    if(os.path.exists(os.path.join(base_dir,o.strftime("%Y%m%d-%H%M%S")+'+'+sub_name))):
                        os.remove(os.path.join(base_dir,o.strftime("%Y%m%d-%H%M%S")+'+'+sub_name))
                    
        print("Save at {}".format(save_dir))
        torch.save(self._net,save_dir)
        return 
        
    def save_opt(self,save_dir=None):
        if(self._opt==None):return
        file_name = "{}_opt.pkl".format(self._task_name if(self._task_name!=None)else "save")
        if(save_dir==None):
            save_dir = os.path.join(self._model_path,file_name)
        elif(len(save_dir.split('.'))==1): 
            save_dir = os.path.join(save_dir,file_name)
        elif(os.path.dirname(save_dir)==''):
            save_dir = os.path.join(self._model_path,save_dir)
        torch.save(self._opt,save_dir)
        print("Save optimizer at {}".format(save_dir))
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
                print("Load at {}.".format(load_dir))
                self._net=torch.load(load_dir)
                self._net=self._net.float().to(self._device)
                try:
                    self._net.init_state()
                except:
                    print("Skip init state function.")
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
        try:
            self._net.init_state()
        except:
            print("Skip init state function.")
        return 

    def log_info(self,info):
        if(self._f_train_loger!=None):
            self._f_train_loger.write(info)
            self._f_train_loger.flush()

    def opt_decay(self,decay_rate=0.9):
        if(self._opt==None):return
        decay_rate = decay_rate if(decay_rate!=None)else 0.9

        if(self._f_train_loger!=None):
            self._f_train_loger.write(
                "Change learning rate form {} to {}.\n".format(
                    self._opt.param_groups[0]['lr'],self._opt.param_groups[0]['lr']*(decay_rate)))
        for param_group in self._opt.param_groups:
            param_group['lr'] *= decay_rate
        return
