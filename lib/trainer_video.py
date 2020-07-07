import torch

from model.craft import CRAFT
from utils.img_hlp import np_img_resize
from trainer import Trainer


net = CRAFT()


net.load_state_dict(copyStateDict(torch.load(args.trained_model)))

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

class CRAFTTrainer(Trainer):
    def __init__(self,use_cuda = True):

        super(CRAFTTrainer,self).__init__(use_cuda=use_cuda)

    def load_weight(self,pth_file):
        if(self.net!=None):
            if(self.use_cuda):
                net.load_state_dict(copyStateDict(torch.load(pth_file)))
            else:
                net.load_state_dict(copyStateDict(torch.load(pth_file, map_location='cpu')))
            

    def test(self, vdo, max_frame = None):
        

    def test_step(self,o):
        if(self.net==None):return
        
        y, feature = net(x)
        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()
        

        



