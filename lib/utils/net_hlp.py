import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import Iterable
from thop import profile
from torchsummary import summary as tsummary

def double_conv(in_ch, mid_ch, out_ch, kernel_size=(1,3), padding=True):
    if(isinstance(kernel_size,Iterable) and len(kernel_size)>=2):
        k1,k2=int(kernel_size[0]),int(kernel_size[1])
    else:
        k1,k2=int(kernel_size),int(kernel_size)
    
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=k1, padding=k2//2 if(padding)else 0),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, kernel_size=k2, padding=k2//2 if(padding)else 0),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

def gen_convs(ch_list:list, kernel_size=3, stride_size=1, padding=True, batchnor_f=nn.BatchNorm2d,activate_f=nn.ReLU):
    if(isinstance(kernel_size,Iterable) and len(kernel_size)<len(ch_list)-1):
        kernel_size = list(kernel_size)+[kernel_size[-1]]*(len(ch_list)-1-len(kernel_size))
    else:
        kernel_size = [kernel_size]*(len(ch_list)-1)
    if(isinstance(stride_size,Iterable) and len(stride_size)<len(ch_list)-1):
        stride_size = list(stride_size)+[stride_size[-1]]*(len(ch_list)-1-len(stride_size))
    else:
        stride_size = [stride_size]*(len(ch_list)-1)
    
    nets = []
    for i in range(len(ch_list)-1):
        nets.append(nn.Conv2d(ch_list[i],ch_list[i+1], kernel_size=kernel_size[i], stride=stride_size[i], padding=kernel_size[i]//2 if(padding)else 0))
        if(batchnor_f):
            nets.append(batchnor_f(ch_list[i+1]))
        if(activate_f):
            nets.append(activate_f(inplace=True))

    return nn.Sequential(*nets)

def conv_bn(in_ch, out_ch, **args):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, **args),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True)
    )
        
def init_weights(modules,method:str = 'xavier_uniform'):
    method = method.lower()
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if('xavier_uniform' in method):
                init.xavier_uniform_(m.weight.data)
            elif('xavier_normal' in method):
                init.xavier_normal_(m.weight.data)
            elif('kaiming_normal' in method):
                init.kaiming_normal_(m.weight, mode='fan_out')
            else:
                m.weight.data.normal_(0, math.sqrt(2. / n))
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

def get_final_ch(modules):
    ch=-1
    for bname in modules.state_dict():
        if('weight' in bname and len(modules.state_dict()[bname].shape)==4):
            ch = modules.state_dict()[bname].shape[0]
    return ch

def get_flops(module, input_size):
    """
    Calculate FLOPs and parameter size for given module and input size
    """
    for k,v in module.state_dict().items():
        d = v
        break
    x = torch.zeros(input_size,dtype=d.dtype,device=d.device)
    flops, params = profile(module, inputs=(x,))
    return flops, params

def summary(module,input_size=(3,640,640)):
    for k,v in module.state_dict().items():
        d = v
        break
    tsummary(module,input_size,device=d.device)

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