from lib.trainer_base import Trainer
from torch.utils.tensorboard import SummaryWriter

class CRAFTTrainer(Trainer):
    def __init__(self,  
        **params
    ):
        Trainer.__init__(self,**params)
    
    def _logger(self,pred,loss,step,batch_size):
        if(self._file_writer!=None):
            writer.add_scalar('Loss/train', Loss, step)
        
        return None
