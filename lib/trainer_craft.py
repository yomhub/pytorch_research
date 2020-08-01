import torch
from lib.trainer_base import Trainer
from torch.utils.tensorboard import SummaryWriter

class CRAFTTrainer(Trainer):
    def __init__(self,  
        **params
    ):
        Trainer.__init__(self,**params)
    
    def _logger(self,x,y,pred,loss,step,batch_size):
        if(self._file_writer==None):return None
        # self._file_writer.add_scalar('Loss/train', loss, step)
        if(len(x.shape)==4):
            self._file_writer.add_image('Image', x[0]/255, step)
        else:
            self._file_writer.add_image('Image', x/255, step)
        char_target, aff_target = y
        if(len(char_target.shape)==4):
            self._file_writer.add_image('Char Gaussian', char_target[0], step)               
        else:
            self._file_writer.add_image('Char Gaussian', char_target, step)

        if(len(aff_target.shape)==4):
            self._file_writer.add_image('Affinity Gaussian', aff_target[0], step)
        else:
            self._file_writer.add_image('Affinity Gaussian', aff_target, step)

        predict_r = torch.reshape(x[:,0,:,:],(x.shape[0],1,x.shape[2],x.shape[3]))
        predict_a = torch.reshape(x[:,1,:,:],(x.shape[0],1,x.shape[2],x.shape[3]))

        if(len(predict_r.shape)==4):
            self._file_writer.add_image('Pred char Gaussian', predict_r[0], step)
        else:
            self._file_writer.add_image('Pred char Gaussian', predict_r, step)

        if(len(predict_a.shape)==4):
            self._file_writer.add_image('Pred affinity Gaussian', predict_a[0], step)
        else:
            self._file_writer.add_image('Pred affinity Gaussian', predict_a, step)

        return None

    def _step_callback(self,x,y,pred,loss,step,batch_size):
        if(self._file_writer==None):return None
        self._file_writer.add_scalar('Loss/train', loss, step)
        return None
