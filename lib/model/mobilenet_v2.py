import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models
import numpy as np
import math
from collections import namedtuple

def double_conv(in_ch, mid_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=1),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
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
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
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
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
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
    def __init__(self, width_mult=1., have_fc=False, n_class=1000):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self._last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
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


def mobilenet_v2(input_size,width_mult,pretrained=True,have_fc=False):
    model = MobileNetV2(input_size=input_size,width_mult=width_mult,have_fc=have_fc)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        if(not have_fc):
            state_dict.pop('classifier.bias')
            state_dict.pop('classifier.weight')
        model.load_state_dict(state_dict)
    return model


class CRAFT_MOB(nn.Module):
    def __init__(self, width_mult=1.,pretrained=False):
        super(CRAFT_MOB, self).__init__()

        inverted_residual_setting = [
            # t, c, n, s
            # 1/2
            [1, 16, 1, 1], # block_0
            [6, 24, 2, 2], # block_1
            # 1/4
            [6, 32, 3, 2], # block_2
            # 1/8
            [6, 64, 4, 2], # block_3
            # 1/16
            [6, 96, 3, 1], # block_4
            [6, 160, 3, 2], # block_5
            # 1/32
            [6, 320, 1, 1], # block_6
        ]
        mob = models.mobilenet_v2(pretrained=True) if(pretrained)else models.mobilenet_v2(
            pretrained=pretrained,
            width_mult=width_mult,
            inverted_residual_setting=inverted_residual_setting)

        self._b1=torch.nn.Sequential()
        self._b2=torch.nn.Sequential()
        self._b3=torch.nn.Sequential()
        self._b4=torch.nn.Sequential()

        self._b0=mob.features[0]
        # self.init_weights(self._b0.modules())
        # self._b0=ConvBNReLU(3,32,norm_layer = nn.BatchNorm2d)

        k,l=1,3
        for i in range(k,k+l):
            # ConvBNReLU, [1, 16, 1, 1],[6, 24, 2, 2]
            self._b1.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        k+=l
        # 1/2

        l=3
        for i in range(k,k+l):
            # [6, 32, 3, 2],
            self._b2.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        k+=l
        # 1/4
        
        l=7
        for i in range(k,k+l):
            # [6, 64, 4, 2],[6, 96, 3, 1]
            self._b3.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        k+=l
        # 1/8

        l=4
        for i in range(k,k+l):
            # [6, 160, 3, 2],[6, 320, 1, 1],ConvBNReLU
            self._b4.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        # 1/16
        k+=l

        out_chs = [make_divisible(c * width_mult)if(t > 1)else c for t, c, n, s in inverted_residual_setting]

        upchs=out_chs[-1]
        # out_chs[-1],[-3]
        self._upc1 = double_conv(upchs+out_chs[-3],(upchs+out_chs[-3])//2,(upchs+out_chs[-3])//2)
        upchs = (upchs+out_chs[-3])//2 #208
        # out_chs[-3],[-5]
        self._upc2 = double_conv(upchs+out_chs[-5],(upchs+out_chs[-5])//2,(upchs+out_chs[-5])//2)
        upchs = (upchs+out_chs[-5])//2 #120
        # out_chs[-5]+[-6]
        self._upc3 = double_conv(upchs+out_chs[-6],(upchs+out_chs[-6])//2,(upchs+out_chs[-6])//2)
        upchs = (upchs+out_chs[-6])//2 #72
        # out_chs[-6]+make_divisible(32 * width_mult)
        inch=make_divisible(32 * width_mult)
        self._upc4 = double_conv(upchs+inch,(upchs+inch)//2,(upchs+inch)//2)
        upchs = (upchs+inch)//2 #52

        self._final_predict = nn.Sequential(
            nn.Conv2d(upchs, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1, stride=1)
        )
        self.init_weights(self._upc1.modules())
        self.init_weights(self._upc2.modules())
        self.init_weights(self._upc3.modules())
        self.init_weights(self._final_predict.modules())

    def init_weights(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_normal_(m.weight.data)
                init.kaiming_normal_(m.weight, mode='fan_out')
                # init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        b0=self._b0(x)
        b1=self._b1(b0)
        b2=self._b2(b1)
        b3=self._b3(b2)
        b4=self._b4(b3)

        upc1 = self._upc1(torch.cat((F.interpolate(b4, size=b3.size()[2:], mode='bilinear', align_corners=False),b3),dim=1))
        upc2 = self._upc2(torch.cat((F.interpolate(upc1, size=b2.size()[2:], mode='bilinear', align_corners=False),b2),dim=1))
        upc3 = self._upc3(torch.cat((F.interpolate(upc2, size=b1.size()[2:], mode='bilinear', align_corners=False),b1),dim=1))
        upc4 = self._upc4(torch.cat((F.interpolate(upc3, size=b0.size()[2:], mode='bilinear', align_corners=False),b0),dim=1))
        pred = self._final_predict(upc4)

        vgg_outputs = namedtuple("MobilenetOuts", ['upc4','upc3','upc2','upc1','b4', 'b3', 'b2', 'b1'])
        vgg_outputs(upc4,upc3,upc2,upc1,b4,b3,b2,b1)

        return pred,vgg_outputs