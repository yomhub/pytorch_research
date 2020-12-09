from collections import namedtuple
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision.models.vgg import model_urls
from lib.utils.net_hlp import init_weights,SoftMaxPool2d

DEF_BASE_NETS = ['vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn']

class VGG(torch.nn.Module):
    def __init__(self, basenet='vgg16_bn', pretrained=True, padding:bool=True, maxpool:bool=True, loadnet:str=None):
        super(VGG, self).__init__()
        basenet = basenet.lower() if(basenet.lower() in DEF_BASE_NETS)else 'vgg16_bn'
        model_urls[basenet] = model_urls[basenet].replace('https://', 'http://')
        if(basenet=='vgg11'):
            vgg_pretrained_features = models.vgg11(pretrained=pretrained).features
        elif(basenet=='vgg11_bn'):
            vgg_pretrained_features = models.vgg11_bn(pretrained=pretrained).features
        elif(basenet=='vgg13'):
            vgg_pretrained_features = models.vgg13(pretrained=pretrained).features
        elif(basenet=='vgg13_bn'):
            vgg_pretrained_features = models.vgg13_bn(pretrained=pretrained).features
        elif(basenet=='vgg16'):
            vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        elif(basenet=='vgg16_bn'):
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
        elif(basenet=='vgg19'):
            vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        else:
            vgg_pretrained_features = models.vgg19_bn(pretrained=pretrained).features
            
        self.b0 = torch.nn.Sequential()
        self.b1 = torch.nn.Sequential()
        self.b2 = torch.nn.Sequential()
        self.b3 = torch.nn.Sequential()
        self.b4 = torch.nn.Sequential()
        init_list = [self.b0,self.b1,self.b2,self.b3,self.b4]
        cnt=0
        for block in init_list:
            while(cnt<len(vgg_pretrained_features)):
                block.add_module(str(cnt), vgg_pretrained_features[cnt])
                if(isinstance(vgg_pretrained_features[cnt],nn.MaxPool2d)):
                    break
                cnt+=1
            cnt+=1


        if(loadnet and os.path.exists(loadnet)):
            para = torch.load(loadnet)
            for o in self.state_dict():
                self.state_dict()[o]=para['module.{}'.format(o)]
        elif(not pretrained):
            for block in init_list:
                init_weights(block.modules())
                init_weights(block.modules())
                init_weights(block.modules())
                init_weights(block.modules())

        if((not padding) or (not maxpool)):
            for m in self.modules():
                if((not padding) and isinstance(m, nn.Conv2d)):
                    m.padding=(0,0)
                if((not maxpool) and isinstance(m,nn.MaxPool2d)):
                    m=SoftMaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1(b0)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)

        vgg_outputs = namedtuple("VggOutputs", ['b0','b1', 'b2', 'b3', 'b4'])
        out = vgg_outputs(b0, b1, b2, b3, b4)
        return out

class VGG16(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True, padding:bool=True, maxpool:bool=True):
        super(VGG16, self).__init__()
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):         # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

        if(freeze):
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad= False
        if((not padding) or (not maxpool)):
            for single in [self.slice1,self.slice2,self.slice3,self.slice4]:
                for m in single.modules():
                    if((not padding) and isinstance(m, nn.Conv2d)):
                        m.padding=(0,0)
                    elif((not maxpool) and isinstance(m,nn.MaxPool2d)):
                        m=SoftMaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out

class VGG16_test(torch.nn.Module):
    def __init__(self, padding:bool=True, maxpool:bool=True, pretrained=True, freeze:bool=True):
        """
        VGG16 bn
        Args:
            padding: enable zero padding for convolution layer
            pretrained: whether to use pretrained network
                file path to load specific pth/pkl file,
                or True to load default torch vgg16
            freeze: whether to freeze B1 network
        Network outs:
            namedtuple:
                'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'
        """
        padding = 1 if(padding)else 0
        mplayer = nn.MaxPool2d if(maxpool)else SoftMaxPool2d
        super(VGG16_test, self).__init__()
        
        # B1+B2
        self.slice1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.BatchNorm2d(64),
            mplayer(kernel_size=(2,2),stride=(2,2),padding=(0,0)),
            nn.Conv2d(64,128,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.Conv2d(128,128,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.BatchNorm2d(128),
            )
        # B3
        self.slice2 = nn.Sequential(
            mplayer(kernel_size=(2,2),stride=(2,2),padding=(0,0)),
            nn.Conv2d(128,256,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.BatchNorm2d(256),
        )
        self.slice3 = nn.Sequential(
            mplayer(kernel_size=(2,2),stride=(2,2),padding=(0,0)),
            nn.Conv2d(256,512,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.BatchNorm2d(512),
        )
        self.slice4 = nn.Sequential(
            mplayer(kernel_size=(2,2),stride=(2,2),padding=(0,0)),
            nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=padding),
            nn.BatchNorm2d(512),
        )
        
        if(pretrained):
            if(isinstance(pretrained,str) and os.path.exists(pretrained)):
                para = torch.load(pretrained)
                try:
                    para = para.state_dict()
                except:
                    para = para
                mdfeatures = [para[o] for o in para]
            else:
                model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
                mdfeatures = models.vgg16_bn(pretrained=pretrained).features
            cnt = 0
            for o in self.slice1.parameters():
                o = mdfeatures[cnt]
                cnt+=1
            for o in self.slice2.parameters():
                o = mdfeatures[cnt]
                cnt+=1
            for o in self.slice3.parameters():
                o = mdfeatures[cnt]
                cnt+=1
            for o in self.slice4.parameters():
                o = mdfeatures[cnt]
                cnt+=1
        else:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h

        vgg_outputs = namedtuple("VggOutputs", ['relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
