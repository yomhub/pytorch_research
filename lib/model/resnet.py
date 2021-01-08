from collections import namedtuple
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models
from torchvision.models.vgg import model_urls
from lib.utils.net_hlp import init_weights,SoftMaxPool2d,get_final_ch,double_conv

DEF_BASE_NETS = ['resnet18','resnet34','resnet101','resnet152']

class Resnet(nn.Module):
    def __init__(self, basenet:str='resnet50', pretrained=True, loadnet:str=None,include_final:bool=False,**args):
        super(Resnet, self).__init__()
        basenet = basenet.lower() if(basenet.lower() in DEF_BASE_NETS)else 'resnet50'

        if(basenet=='resnet18'):
            base_cls = models.resnet18
        elif(basenet=='resnet34'):
            base_cls = models.resnet34
        elif(basenet=='resnet101'):
            base_cls = models.resnet101
        elif(basenet=='resnet152'):
            base_cls = models.resnet152
        else:
            base_cls = models.resnet50
        
        basnet = base_cls(pretrained=bool(pretrained))
        
        self.b0 = torch.nn.Sequential(basnet.conv1,basnet.bn1,basnet.relu,basnet.maxpool)
        self.b1 = basnet.layer1
        self.b2 = basnet.layer2
        self.b3 = basnet.layer3
        self.b4 = basnet.layer4
        init_list = [self.b0,self.b1,self.b2,self.b3,self.b4]

        self.include_final = bool(include_final)
        if(self.include_final):
            self.b5 = torch.nn.Sequential(
                nn.Conv2d(get_final_ch(self.b4), 2048, kernel_size=1),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
            )
            init_list.append(self.b5)

        if(loadnet and os.path.exists(loadnet)):
            para = torch.load(loadnet)
            for o in self.state_dict():
                self.state_dict()[o]=para['module.{}'.format(o)]
        elif(not pretrained):
            for block in init_list:
                init_weights(block.modules())
        
        self.output_tuple = namedtuple("ResnetOutputs", ['b{}'.format(i) for i in range(len(init_list))])

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1(b0)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        if(self.include_final):
            b5 = self.b5(b4)
            return self.output_tuple(b0, b1, b2, b3, b4, b5)
        return self.output_tuple(b0, b1, b2, b3, b4)

class ResnetUnet(nn.Module):
    def __init__(self, min_upc_ch:int=0,
        init_method:str='xavier_uniform',include_final:bool=False,**args):
        super(ResnetUnet, self).__init__()
        self.basenet = Resnet(include_final=include_final,**args)

        b4ch = get_final_ch(self.basenet.b4)
        b3ch = get_final_ch(self.basenet.b3)
        b2ch = get_final_ch(self.basenet.b2)
        b1ch = get_final_ch(self.basenet.b1)
        b0ch = get_final_ch(self.basenet.b0)

        self.include_final = bool(include_final)
        if(self.include_final):
            b5ch = get_final_ch(self.basenet.b5)
            self.b5b4_b4 = double_conv(b4ch+b5ch,(b4ch+b5ch)//2,b4ch)
            init_weights(self.b5b4_b4.modules(),init_method)

        b4b3_b3_out = max(min_upc_ch,b3ch)
        b3b2_b2_out = max(min_upc_ch,b2ch)
        b2b1_b1_out = max(min_upc_ch,b1ch)
        b4b3_b3_ct = max(b4b3_b3_out,(b4ch+b3ch)//2)
        b3b2_b2_ct = max(b3b2_b2_out,(b4b3_b3_out+b2ch)//2)
        b2b1_b1_ct = max(b2b1_b1_out,(b3b2_b2_out+b1ch)//2)

        self.b4b3_b3 = double_conv(b4ch+b3ch,b4b3_b3_ct,b4b3_b3_out)
        self.b3b2_b2 = double_conv(b4b3_b3_out+b2ch,b3b2_b2_ct,b3b2_b2_out)
        self.b2b1_b1 = double_conv(b3b2_b2_out+b1ch,b2b1_b1_ct,b2b1_b1_out)
        self.out_channels = b2b1_b1_out+b0ch
        out_tuples = ['upb0','upb1','upb2','upb3','b0','b1', 'b2', 'b3', 'b4']
        if(self.include_final):
            out_tuples.append('b5')
        self.output_tuple = namedtuple("VggUnetOutputs", out_tuples)
        init_weights(self.b4b3_b3.modules(),init_method)
        init_weights(self.b3b2_b2.modules(),init_method)
        init_weights(self.b2b1_b1.modules(),init_method)
        
    def forward(self,x):
        vgg_feat = self.basenet(x)
        feat = vgg_feat.b4
        if(self.include_final):
            feat = self.b5b4_b4(torch.cat((vgg_feat.b5,vgg_feat.b4),1))
        feat = torch.cat((vgg_feat.b3,F.interpolate(feat,vgg_feat.b3.shape[2:], mode='bilinear', align_corners=False)),dim=1)
        feat = self.b4b3_b3(feat)
        upb3 = feat
        feat = torch.cat((vgg_feat.b2,F.interpolate(feat,vgg_feat.b2.shape[2:], mode='bilinear', align_corners=False)),dim=1)
        feat = self.b3b2_b2(feat)
        upb2 = feat
        feat = torch.cat((vgg_feat.b1,F.interpolate(feat,vgg_feat.b1.shape[2:], mode='bilinear', align_corners=False)),dim=1)
        feat = self.b2b1_b1(feat)
        upb1 = feat

        upb0 = torch.cat((vgg_feat.b0,F.interpolate(feat,vgg_feat.b0.shape[2:], mode='bilinear', align_corners=False)),dim=1)
        if(self.include_final):
            ufeat = self.output_tuple(upb0,upb1,upb2,upb3,vgg_feat.b0,vgg_feat.b1,vgg_feat.b2,vgg_feat.b3,vgg_feat.b4,vgg_feat.b5)
        else:
            ufeat = self.output_tuple(upb0,upb1,upb2,upb3,vgg_feat.b0,vgg_feat.b1,vgg_feat.b2,vgg_feat.b3,vgg_feat.b4)

        return ufeat