import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.model.mobilenet_v2 import *
from collections import namedtuple
from lib.model.vgg16 import VGG,VGGUnet
from lib.utils.net_hlp import get_final_ch

class PIX_TXT(nn.Module):
    def __init__(self,
        map_ch = 3,
        width_mult=1.,pretrained=False,
        load_mobnet:str=None,
        inverted_residual_setting = [
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
            ], **args):
        super(PIX_TXT, self).__init__()

        mob = models.mobilenet_v2(
            pretrained=pretrained,
            width_mult=width_mult,
            inverted_residual_setting=inverted_residual_setting)

        # blocks
        self.b0=mob.features[0]
        self.b1=torch.nn.Sequential()
        self.b2=torch.nn.Sequential()
        self.b3=torch.nn.Sequential()
        self.b4=torch.nn.Sequential()

        # copy the parameters to block
        k,l=1,3
        for i in range(k,k+l):
            # ConvBNReLU, [1, 16, 1, 1],[6, 24, 2, 2]
            self.b1.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        k+=l
        # 1/2

        l=3
        for i in range(k,k+l):
            # [6, 32, 3, 2],
            self.b2.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        k+=l
        # 1/4
        
        l=7
        for i in range(k,k+l):
            # [6, 64, 4, 2],[6, 96, 3, 1]
            self.b3.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        k+=l
        # 1/8

        l=4
        for i in range(k,k+l):
            # [6, 160, 3, 2],[6, 320, 1, 1],ConvBNReLU
            self.b4.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        # 1/16
        k+=l

        out_chs = [make_divisible(c * width_mult)if(t > 1)else c for t, c, n, s in inverted_residual_setting]

        self.b4_mask = nn.Sequential(
            double_conv(out_chs[-1],out_chs[-1]//2,out_chs[-1]//4),
            double_conv(out_chs[-1]//4,out_chs[-1]//8,map_ch),
            )
        self.b3_mask = nn.Sequential(
            double_conv(out_chs[-3],out_chs[-3]//2,out_chs[-3]//4),
            double_conv(out_chs[-3]//4,out_chs[-3]//8,map_ch),
            )
        self.b2_mask = nn.Sequential(
            double_conv(out_chs[-5],out_chs[-5]//2,out_chs[-5]//4),
            double_conv(out_chs[-5]//4,out_chs[-5]//8,map_ch),
            )
        self.b1_mask = nn.Sequential(
            double_conv(out_chs[-6],out_chs[-6]//2,out_chs[-6]//4),
            double_conv(out_chs[-6]//4,out_chs[-6]//8,map_ch),
            )
        if(pretrained):
            self.init_weights(self.b4_mask.modules())
            self.init_weights(self.b3_mask.modules())
            self.init_weights(self.b2_mask.modules())
            self.init_weights(self.b1_mask.modules())
        else:
            self.init_weights(self.modules())
        if(load_mobnet and os.path.exists(load_mobnet)):
            para = torch.load(load_mobnet)
            for o in self.state_dict():
                if('mask' not in o):
                    self.state_dict()[o]=para['module.{}'.format(o)]
    def init_weights(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_normal_(m.weight.data)
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        b0=self.b0(x)
        b1=self.b1(b0)
        b2=self.b2(b1)
        b3=self.b3(b2)
        b4=self.b4(b3)
        b4_mask = self.b4_mask(b4)
        b3_mask = self.b3_mask(b3)
        b2_mask = self.b2_mask(b2)
        b1_mask = self.b1_mask(b1)
        mask = b4_mask
        mask = b3_mask+F.interpolate(mask, size=b3_mask.size()[2:], mode='bilinear', align_corners=False)
        mask = b2_mask+F.interpolate(mask, size=b2_mask.size()[2:], mode='bilinear', align_corners=False)
        mask = b1_mask+F.interpolate(mask, size=b1_mask.size()[2:], mode='bilinear', align_corners=False)
        feat = namedtuple("pixelmap", ['b1_mask', 'b2_mask', 'b3_mask', 'b4_mask', 'b1','b2','b3','b4'])
        featout = feat(b1_mask, b2_mask, b3_mask, b4_mask, b1,b2,b3,b4)
        return mask,featout

class PIX_MASK(nn.Module):
    def __init__(self,width_mult=1.,pretrained=False,
        load_mobnet:str=None,
        inverted_residual_setting = [
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
            ], **args):
        super(PIX_MASK, self).__init__()

        mob = models.mobilenet_v2(
            pretrained=pretrained,
            width_mult=width_mult,
            inverted_residual_setting=inverted_residual_setting)

        # blocks
        self.b0=mob.features[0]
        self.b1=torch.nn.Sequential()
        self.b2=torch.nn.Sequential()
        self.b3=torch.nn.Sequential()
        self.b4=torch.nn.Sequential()

        # copy the parameters to block
        k,l=1,3
        for i in range(k,k+l):
            # ConvBNReLU, [1, 16, 1, 1],[6, 24, 2, 2]
            self.b1.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        k+=l
        # 1/2

        l=3
        for i in range(k,k+l):
            # [6, 32, 3, 2],
            self.b2.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        k+=l
        # 1/4
        
        l=7
        for i in range(k,k+l):
            # [6, 64, 4, 2],[6, 96, 3, 1]
            self.b3.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        k+=l
        # 1/8

        l=4
        for i in range(k,k+l):
            # [6, 160, 3, 2],[6, 320, 1, 1],ConvBNReLU
            self.b4.add_module(str(i)+mob.features[i].__class__.__name__,mob.features[i])
        # 1/16
        k+=l

        out_chs = [make_divisible(c * width_mult)if(t > 1)else c for t, c, n, s in inverted_residual_setting]
        map_ch = 3
        # bg fg txt
        self.b4_mask = nn.Sequential(
            double_conv(out_chs[-1],out_chs[-1]//2,max(out_chs[-1]//4,32)),
            double_conv(max(out_chs[-1]//4,32),max(out_chs[-1]//8,16),3),
            )
        # bg fg txt
        self.b3_mask = nn.Sequential(
            double_conv(out_chs[-3],out_chs[-3]//2,max(out_chs[-3]//4,32)),
            double_conv(max(out_chs[-3]//4,32),max(out_chs[-3]//8,16),3),
            )
        # txt edge
        self.b2_mask = nn.Sequential(
            double_conv(out_chs[-5],out_chs[-5]//2,max(out_chs[-5]//4,32)),
            double_conv(max(out_chs[-5]//4,32),max(out_chs[-5]//8,16),2),
            )
        # txt edge
        self.b1_mask = nn.Sequential(
            double_conv(out_chs[-6],max(out_chs[-6]//2,32),max(out_chs[-6]//4,32)),
            double_conv(max(out_chs[-6]//4,32),max(out_chs[-6]//8,16),2),
            )

        cls_num = 3
        # bg region boundary
        self.b4_cls = nn.Sequential(
            double_conv(out_chs[-1],out_chs[-1]//2,max(out_chs[-1]//4,32)),
            double_conv(max(out_chs[-1]//4,32),max(out_chs[-1]//8,16),cls_num),
            )
        self.b3_cls = nn.Sequential(
            double_conv(out_chs[-3],out_chs[-3]//2,max(out_chs[-3]//4,32)),
            double_conv(max(out_chs[-3]//4,32),max(out_chs[-3]//8,16),cls_num),
            )
        self.b2_cls = nn.Sequential(
            double_conv(out_chs[-5],out_chs[-5]//2,max(out_chs[-5]//4,32)),
            double_conv(max(out_chs[-5]//4,32),max(out_chs[-5]//8,16),cls_num),
            )
        self.b1_cls = nn.Sequential(
            double_conv(out_chs[-6],max(out_chs[-6]//2,32),max(out_chs[-6]//4,32)),
            double_conv(max(out_chs[-6]//4,32),max(out_chs[-6]//8,16),cls_num),
            )
        if(pretrained):
            self.init_weights(self.b4_mask.modules())
            self.init_weights(self.b3_mask.modules())
            self.init_weights(self.b2_mask.modules())
            self.init_weights(self.b1_mask.modules())
            self.init_weights(self.b4_cls.modules())
            self.init_weights(self.b3_cls.modules())
            self.init_weights(self.b2_cls.modules())
            self.init_weights(self.b1_cls.modules())
        else:
            self.init_weights(self.modules())
        if(load_mobnet and os.path.exists(load_mobnet)):
            para = torch.load(load_mobnet)
            for o in self.state_dict():
                if('mask' not in o and 'cls' not in o):
                    self.state_dict()[o]=para['module.{}'.format(o)]
    def init_weights(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_normal_(m.weight.data)
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        b0=self.b0(x)
        b1=self.b1(b0)
        b2=self.b2(b1)
        b3=self.b3(b2)
        b4=self.b4(b3)
        # bg fg txt
        b4_mask = self.b4_mask(b4)
        b3_mask = self.b3_mask(b3)
        # txt,edge
        b2_mask = self.b2_mask(b2)
        b1_mask = self.b1_mask(b1)
        b4_mask = F.interpolate(b4_mask, size=b1_mask.size()[2:], mode='bilinear', align_corners=False)
        b3_mask = F.interpolate(b3_mask, size=b1_mask.size()[2:], mode='bilinear', align_corners=False)
        b2_mask = F.interpolate(b2_mask, size=b1_mask.size()[2:], mode='bilinear', align_corners=False)
        edg_mask = F.elu(F.elu(F.elu(b1_mask[:,1])+b2_mask[:,1])-b3_mask[:,0])-b4_mask[:,0]
        txt_mask = F.elu(F.elu(F.elu(F.elu(b1_mask[:,0])+b2_mask[:,0])+b3_mask[:,2]-b3_mask[:,0])+b4_mask[:,2]-b4_mask[:,0])
        region_mask = F.elu(F.elu(b3_mask[:,1]-b3_mask[:,0])+b4_mask[:,1]-b4_mask[:,0])
        mask = torch.stack((txt_mask,edg_mask,region_mask),1)
        b4_cls = self.b4_cls(b4)
        b3_cls = self.b3_cls(b3)
        b2_cls = self.b2_cls(b2)
        b1_cls = self.b1_cls(b1)
        f_cls = b4_cls
        f_cls = b3_cls+F.interpolate(f_cls, size=b3_cls.size()[2:], mode='bilinear', align_corners=False)
        f_cls = b2_cls+F.interpolate(f_cls, size=b2_cls.size()[2:], mode='bilinear', align_corners=False)
        f_cls = b1_cls+F.interpolate(f_cls, size=b1_cls.size()[2:], mode='bilinear', align_corners=False)
        mask = torch.cat((mask,f_cls),1)
        feat = namedtuple("pixelmap", ['b1_mask', 'b2_mask', 'b3_mask', 'b4_mask', 'b1','b2','b3','b4'])
        featout = feat(b1_mask, b2_mask, b3_mask, b4_mask, b1,b2,b3,b4)
        return mask,featout

class VGG_PXMASK(nn.Module):
    def __init__(self,padding:bool=True,**args):
        super(VGG_PXMASK, self).__init__()
        self.basenet = VGG(padding=padding,**args)
        ch0 = get_final_ch(self.basenet.b0)
        ch1 = get_final_ch(self.basenet.b1)
        ch2 = get_final_ch(self.basenet.b2)
        ch3 = get_final_ch(self.basenet.b3)
        ch4 = get_final_ch(self.basenet.b4)
        map_ch = 3
        self.b0_mask = nn.Sequential(
            double_conv(ch0,ch0//2,ch0//4,padding),
            double_conv(ch0//4,ch0//4,map_ch,padding),
            )
        self.b1_mask = nn.Sequential(
            double_conv(ch1,ch1//2,ch1//4,padding),
            double_conv(ch1//4,ch1//8,map_ch,padding),
            )
        self.b2_mask = nn.Sequential(
            double_conv(ch2,ch2//2,ch2//4,padding),
            double_conv(ch2//4,ch2//8,map_ch,padding),
            )
        self.b3_mask = nn.Sequential(
            double_conv(ch3,ch3//2,ch3//4,padding),
            double_conv(ch3//4,ch3//8,map_ch,padding),
            )
        self.b4_mask = nn.Sequential(
            double_conv(ch4,ch4//2,ch4//4,padding),
            double_conv(ch4//4,ch4//8,map_ch,padding),
            )

        self.init_weights(self.b4_mask.modules())
        self.init_weights(self.b3_mask.modules())
        self.init_weights(self.b2_mask.modules())
        self.init_weights(self.b1_mask.modules())
        self.init_weights(self.b0_mask.modules())

    def init_weights(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_normal_(m.weight.data)
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        b0, b1, b2, b3, b4 = self.basenet(x)

        b4_mask = self.b4_mask(b4)
        b3_mask = self.b3_mask(b3)
        b2_mask = self.b2_mask(b2)
        b1_mask = self.b1_mask(b1)
        b0_mask = self.b0_mask(b0)
        mask = b4_mask
        mask = b3_mask+F.interpolate(mask, size=b3_mask.size()[2:], mode='bilinear', align_corners=False)
        mask = b2_mask+F.interpolate(mask, size=b2_mask.size()[2:], mode='bilinear', align_corners=False)
        mask = b1_mask+F.interpolate(mask, size=b1_mask.size()[2:], mode='bilinear', align_corners=False)
        mask = b0_mask+F.interpolate(mask, size=b0_mask.size()[2:], mode='bilinear', align_corners=False)
        feat = namedtuple("pixelmap", ['b0_mask','b1_mask', 'b2_mask', 'b3_mask', 'b4_mask', 'b1','b2','b3','b4'])
        featout = feat(b0_mask, b1_mask, b2_mask, b3_mask, b4_mask, b1,b2,b3,b4)
        return mask,featout

class VGGUnet_PXMASK(nn.Module):
    def __init__(self,padding:bool=True,**args):
        super(VGGUnet_PXMASK, self).__init__()
        self.basenet = VGGUnet(padding=padding,**args)
        outch = self.basenet.final_ch
        self.final_mask = nn.Sequential(
            double_conv(outch,128,64,padding=padding),
            double_conv(64,32,3,padding=padding),
            )
        self.init_weights(self.final_mask.modules())

    def init_weights(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_normal_(m.weight.data)
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        unet_feat,vgg_feat = self.basenet(x)

        mask = self.final_mask(unet_feat)
        
        return mask,unet_feat

class VGG_PUR_CLS(nn.Module):
    def __init__(self,padding:bool=True,**args):
        super(VGG_PUR_CLS, self).__init__()
        self.basenet = VGG(padding=padding,**args)
        b4ch = get_final_ch(self.basenet.b4)
        b3ch = get_final_ch(self.basenet.b3)
        b2ch = get_final_ch(self.basenet.b2)
        b1ch = get_final_ch(self.basenet.b1)
        b0ch = get_final_ch(self.basenet.b0)

        cls_num = 3
        # bg region boundary
        self.b4_cls = nn.Sequential(
            double_conv(b4ch,b4ch//2,max(b4ch//4,32),padding=padding),
            double_conv(max(b4ch//4,32),max(b4ch//4,16),cls_num),
            )
        self.b3_cls = nn.Sequential(
            double_conv(b3ch,b3ch//2,max(b3ch//4,32),padding=padding),
            double_conv(max(b3ch//4,32),max(b3ch//4,16),cls_num),
            )
        self.b2_cls = nn.Sequential(
            double_conv(b2ch,b2ch//2,max(b2ch//4,32),padding=padding),
            double_conv(max(b2ch//4,32),max(b2ch//4,16),cls_num),
            )
        self.b1_cls = nn.Sequential(
            double_conv(b1ch,b1ch//2,max(b1ch//4,32),padding=padding),
            double_conv(max(b1ch//4,32),max(b1ch//4,16),cls_num),
            )
        self.b0_cls = nn.Sequential(
            double_conv(b0ch,b0ch//2,max(b0ch//4,32),padding=padding),
            double_conv(max(b0ch//4,32),max(b0ch//4,16),cls_num),
            )
        self.init_weights(self.b4_cls.modules())
        self.init_weights(self.b3_cls.modules())
        self.init_weights(self.b2_cls.modules())
        self.init_weights(self.b1_cls.modules())
        self.init_weights(self.b0_cls.modules())

    def init_weights(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_normal_(m.weight.data)
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        vgg_feat = self.basenet(x)
        b0, b1, b2, b3, b4 = vgg_feat
        b4_clsp = self.b4_cls(b4)
        b3_clsp = self.b3_cls(b3)
        b2_clsp = self.b2_cls(b2)
        b1_clsp = self.b1_cls(b1)
        b0_clsp = self.b0_cls(b0)
        final_cls = b4_clsp
        final_cls = b3_clsp+F.interpolate(final_cls, size=b3_clsp.size()[2:], mode='nearest')
        final_cls = b2_clsp+F.interpolate(final_cls, size=b2_clsp.size()[2:], mode='nearest')
        final_cls = b1_clsp+F.interpolate(final_cls, size=b1_clsp.size()[2:], mode='nearest')
        final_cls = b0_clsp+F.interpolate(final_cls, size=b0_clsp.size()[2:], mode='nearest')
        return final_cls,vgg_feat