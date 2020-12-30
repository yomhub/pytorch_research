import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models
import numpy as np
import math
from lib.utils.net_hlp import get_final_ch,init_weights
from collections import namedtuple,Iterable

DEF_INTERVERTED_RESIDUAL_SETTING = [
    # t, c, n, s
    # initial doun sample
    # 1/2
    [1, 16, 1, 1], # block_
    [6, 24, 2, 2], # block_1
    # 1/4
    [6, 32, 3, 2], # block_2
    # 1/8
    [6, 64, 4, 2], # block_3
    # 1/16
    [6, 96, 3, 1], # block_
    [6, 160, 3, 2], # block_4
    # 1/32
    [6, 320, 1, 1], # block_6
    ]

def double_conv(in_ch, mid_ch, out_ch, padding=True):
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=1),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1 if(padding)else 0),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

def conv_bn(inp, oup, stride,padding=True):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1 if(padding)else 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, padding=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup and padding

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1 if(padding)else 0, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1 if(padding)else 0, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1., 
        interverted_residual_setting = DEF_INTERVERTED_RESIDUAL_SETTING,
        have_fc=False, n_class=1000,padding=True):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280


        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self._last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, padding)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t,padding=padding))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t,padding=padding))
                input_channel = output_channel
        
        # building classifier
        self._have_fc = bool(have_fc)
        if(self._have_fc):
            # building last several layers
            self.features.append(conv_1x1_bn(input_channel, self._last_channel))
            self.classifier = nn.Linear(self._last_channel, n_class)
        else:
            self._last_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        if(self._have_fc):
            x = x.mean(3).mean(2)
            x = self.classifier(x)
        return x

    def output_size(self):
        return self._last_channel
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MobNetBlk(nn.Module):
    def __init__(self, width_mult=1.,pretrained=False,
        inverted_residual_setting = DEF_INTERVERTED_RESIDUAL_SETTING,
        include_final:bool=False,**args
        ):
        super(MobNetBlk, self).__init__()
        mob = models.mobilenet_v2(
            pretrained=bool(pretrained),
            width_mult=width_mult,
            inverted_residual_setting=inverted_residual_setting)

        # blocks
        self.b0=torch.nn.Sequential()
        self.b1=torch.nn.Sequential()
        self.b2=torch.nn.Sequential()
        self.b3=torch.nn.Sequential()
        self.b4=torch.nn.Sequential()
        self.b0.add_module(str(0)+mob.features[0].__class__.__name__,mob.features[0])
        blk_lst = [self.b0,self.b1,self.b2,self.b3,self.b4]
        blk_idx,feat_idx=0,1
        for t, c, n, s in inverted_residual_setting:
            if(s>1):
                # down sampling, go to next block
                blk_idx = min(blk_idx+1,len(blk_lst)-1)
            for i in range(n):
                blk_lst[blk_idx].add_module(
                    str(feat_idx+i)+mob.features[feat_idx+i].__class__.__name__,mob.features[feat_idx+i])
            feat_idx+=n
        self.include_final = bool(include_final)
        if(self.include_final):
            self.b5=torch.nn.Sequential()
            blk_lst.append(self.b5)
            blk_lst[-1].add_module(str(feat_idx)+mob.features[feat_idx].__class__.__name__,mob.features[feat_idx])

        self.out_tuple = namedtuple("MobNetBlk", ['b{}'.format(i) for i in range(len(blk_lst))])

    def forward(self,x):
        b0 = self.b0(x)
        b1 = self.b1(b0)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        if(self.include_final):
            b5 = self.b5(b4)
            return self.out_tuple(b0,b1,b2,b3,b4,b5)
        return self.out_tuple(b0,b1,b2,b3,b4)

class MobUNet(nn.Module):
    def __init__(self,min_upc_ch:int=128,init_method:str='xavier_uniform',**args):
        super(MobUNet, self).__init__()

        self.basenet = MobNetBlk(**args)
        
        b0ch = get_final_ch(self.basenet.b0)
        b1ch = get_final_ch(self.basenet.b1)
        b2ch = get_final_ch(self.basenet.b2)
        b3ch = get_final_ch(self.basenet.b3)
        b4ch = get_final_ch(self.basenet.b4)
        if(min_upc_ch==None):
            min_upc_ch=0
        b4b3_b3_out = max(min_upc_ch,b3ch)
        b3b2_b2_out = max(min_upc_ch,b2ch)
        b2b1_b1_out = max(min_upc_ch,b1ch)
        b1b0_b0_out = max(min_upc_ch,b0ch)
        b4b3_b3_ct = max(b4b3_b3_out,(b4ch+b3ch)//2)
        b3b2_b2_ct = max(b3b2_b2_out,(b4b3_b3_out+b2ch)//2)
        b2b1_b1_ct = max(b2b1_b1_out,(b3b2_b2_out+b1ch)//2)
        b1b0_b0_ct = max(b1b0_b0_out,(b2b1_b1_out+b0ch)//2)
        self.b4b3_b3 = double_conv(b4ch+b3ch,b4b3_b3_ct,b4b3_b3_out,padding=True)
        self.b3b2_b2 = double_conv(b4b3_b3_out+b2ch,b3b2_b2_ct,b3b2_b2_out,padding=True)
        self.b2b1_b1 = double_conv(b3b2_b2_out+b1ch,b2b1_b1_ct,b2b1_b1_out,padding=True)
        self.b1b0_b0 = double_conv(b2b1_b1_out+b0ch,b1b0_b0_ct,b1b0_b0_out,padding=True)

        self.out_channels = b1b0_b0_out
        self.out_tuple = namedtuple("MobUNet", ['upb0','upb1','upb2','upb3','b0','b1','b2','b3','b4'])
        
        init_weights(self.b4b3_b3.modules(),init_method)
        init_weights(self.b3b2_b2.modules(),init_method)
        init_weights(self.b2b1_b1.modules(),init_method)
        init_weights(self.b1b0_b0.modules(),init_method)
        
    def forward(self,x):
        feat = self.basenet(x)
        upb3 = self.b4b3_b3(torch.cat((F.interpolate(feat.b4, size=feat.b3.size()[2:], mode='bilinear', align_corners=False),feat.b3),dim=1))
        upb2 = self.b3b2_b2(torch.cat((F.interpolate(upb3, size=feat.b2.size()[2:], mode='bilinear', align_corners=False),feat.b2),dim=1))
        upb1 = self.b2b1_b1(torch.cat((F.interpolate(upb2, size=feat.b1.size()[2:], mode='bilinear', align_corners=False),feat.b1),dim=1))
        upb0 = self.b1b0_b0(torch.cat((F.interpolate(upb1, size=feat.b0.size()[2:], mode='bilinear', align_corners=False),feat.b0),dim=1))

        return self.out_tuple(upb0,upb1,upb2,upb3,feat.b0,feat.b1,feat.b2,feat.b3,feat.b4)
