import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import Iterable

def double_conv(in_ch, mid_ch, out_ch, padding=True):
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=1),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1 if(padding)else 0),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
        
def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

def adjust_learning_rate(optimizer, gamma):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= gamma

class Swish_act(nn.Module):
    def __init__(self):
        super(Swish_act, self).__init__()
 
    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class SoftMaxPool2d(nn.Module):
    def __init__(self, kernel_size=(2,2), stride=(2,2), padding=(0,0)):
        super(SoftMaxPool2d, self).__init__()
        if(not isinstance(kernel_size,Iterable)):
            kernel_size = (kernel_size,kernel_size)
        if(not isinstance(stride,Iterable)):
            stride = (stride,stride)
        if(not isinstance(padding,Iterable)):
            padding = (padding,padding)
        self.kernel_size = kernel_size
        self.stride=stride
        self.padding=padding
        self.mp = nn.Sequential(
            nn.MaxPool2d(kernel_size=(kernel_size[0],1),stride=(stride[0],1),padding=(padding[0],0)),
            nn.ZeroPad2d((padding[1]//2,padding[1]-(padding[1]//2),0,0))
            )

    def forward(self, x):
        x = self.mp(x)

        x = F.interpolate(x,size=(x.shape[-2],(x.shape[-1])//self.stride[-1]), mode='bilinear', align_corners=False)
        return x