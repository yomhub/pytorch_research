import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

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