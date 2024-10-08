import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.model.mobilenet_v2 import *
from collections import namedtuple
from lib.model.vgg16 import VGG,VGGUnet
from lib.model.resnet import Resnet,ResnetUnet
from lib.model.lstm import BottleneckLSTMCell
from lib.utils.net_hlp import get_final_ch,init_weights

class PIX_TXT(nn.Module):
    def __init__(self,map_ch = 3, min_map_ch:int=32, 
        init_method:str='xavier_uniform',**args):
        super(PIX_TXT, self).__init__()

        self.basenet = MobNetBlk(**args)

        b0ch = get_final_ch(self.basenet.b0)
        b1ch = get_final_ch(self.basenet.b1)
        b2ch = get_final_ch(self.basenet.b2)
        b3ch = get_final_ch(self.basenet.b3)
        b4ch = get_final_ch(self.basenet.b4)

        self.b4_mask = nn.Sequential(
            double_conv(b4ch,max(b4ch//2,min_map_ch),max(b4ch//4,min_map_ch)),
            double_conv(max(b4ch//4,min_map_ch),max(b4ch//8,min_map_ch),map_ch),
            )
        self.b3_mask = nn.Sequential(
            double_conv(b3ch,max(b3ch//2,min_map_ch),max(b3ch//4,min_map_ch)),
            double_conv(max(b3ch//4,min_map_ch),max(b3ch//8,min_map_ch),map_ch),
            )
        self.b2_mask = nn.Sequential(
            double_conv(b2ch,max(b2ch//2,min_map_ch),max(b2ch//4,min_map_ch)),
            double_conv(max(b2ch//4,min_map_ch),max(b2ch//8,min_map_ch),map_ch),
            )
        self.b1_mask = nn.Sequential(
            double_conv(b1ch,max(b1ch//2,min_map_ch),max(b1ch//4,min_map_ch)),
            double_conv(max(b1ch//4,min_map_ch),max(b1ch//8,min_map_ch),map_ch),
            )

        self.out_tuple = namedtuple("PIX_TXT", ['b1_mask', 'b2_mask', 'b3_mask', 'b4_mask', 'b1','b2','b3','b4'])
        init_weights(self.b4_mask.modules(),init_method)
        init_weights(self.b3_mask.modules(),init_method)
        init_weights(self.b2_mask.modules(),init_method)
        init_weights(self.b1_mask.modules(),init_method)

    def forward(self,x):
        feat=self.basenet(x)
        b4_mask = self.b4_mask(feat.b4)
        b3_mask = self.b3_mask(feat.b3)
        b2_mask = self.b2_mask(feat.b2)
        b1_mask = self.b1_mask(feat.b1)
        mask = b4_mask
        mask = b3_mask+F.interpolate(mask, size=b3_mask.size()[2:], mode='bilinear', align_corners=False)
        mask = b2_mask+F.interpolate(mask, size=b2_mask.size()[2:], mode='bilinear', align_corners=False)
        mask = b1_mask+F.interpolate(mask, size=b1_mask.size()[2:], mode='bilinear', align_corners=False)

        featout = self.out_tuple(b1_mask, b2_mask, b3_mask, b4_mask, feat.b1,feat.b2,feat.b3,feat.b4)
        return mask,featout

class PIX_MASK(nn.Module):
    def __init__(self, min_map_ch:int=32, init_method:str='xavier_uniform', **args):
        super(PIX_MASK, self).__init__()
        self.basenet = MobNetBlk(**args)

        b0ch = get_final_ch(self.basenet.b0)
        b1ch = get_final_ch(self.basenet.b1)
        b2ch = get_final_ch(self.basenet.b2)
        b3ch = get_final_ch(self.basenet.b3)
        b4ch = get_final_ch(self.basenet.b4)
        
        map_ch = 3
        # bg fg txt
        self.b4_mask = nn.Sequential(
            double_conv(b4ch,max(b4ch//2,min_map_ch),max(b4ch//4,min_map_ch)),
            double_conv(max(b4ch//4,min_map_ch),max(b4ch//8,min_map_ch),map_ch),
            )
        self.b3_mask = nn.Sequential(
            double_conv(b3ch,max(b3ch//2,min_map_ch),max(b3ch//4,min_map_ch)),
            double_conv(max(b3ch//4,min_map_ch),max(b3ch//8,min_map_ch),map_ch),
            )
        # txt edge
        self.b2_mask = nn.Sequential(
            double_conv(b2ch,max(b2ch//2,min_map_ch),max(b2ch//4,min_map_ch)),
            double_conv(max(b2ch//4,min_map_ch),max(b2ch//8,min_map_ch),map_ch),
            )
        self.b1_mask = nn.Sequential(
            double_conv(b1ch,max(b1ch//2,min_map_ch),max(b1ch//4,min_map_ch)),
            double_conv(max(b1ch//4,min_map_ch),max(b1ch//8,min_map_ch),map_ch),
            )

        cls_ch = 3
        self.b4_cls = nn.Sequential(
            double_conv(b4ch,max(b4ch//2,min_map_ch),max(b4ch//4,min_map_ch)),
            double_conv(max(b4ch//4,min_map_ch),max(b4ch//8,min_map_ch),cls_ch),
            )
        self.b3_cls = nn.Sequential(
            double_conv(b3ch,max(b3ch//2,min_map_ch),max(b3ch//4,min_map_ch)),
            double_conv(max(b3ch//4,min_map_ch),max(b3ch//8,min_map_ch),cls_ch),
            )
        # txt edge
        self.b2_cls = nn.Sequential(
            double_conv(b2ch,max(b2ch//2,min_map_ch),max(b2ch//4,min_map_ch)),
            double_conv(max(b2ch//4,min_map_ch),max(b2ch//8,min_map_ch),cls_ch),
            )
        self.b1_cls = nn.Sequential(
            double_conv(b1ch,max(b1ch//2,min_map_ch),max(b1ch//4,min_map_ch)),
            double_conv(max(b1ch//4,min_map_ch),max(b1ch//8,min_map_ch),cls_ch),
            )

        self.out_tuple = namedtuple("PIX_MASK", ['b1_mask', 'b2_mask', 'b3_mask', 'b4_mask', 'b1','b2','b3','b4'])
        init_weights(self.b4_mask.modules(),init_method)
        init_weights(self.b3_mask.modules(),init_method)
        init_weights(self.b2_mask.modules(),init_method)
        init_weights(self.b1_mask.modules(),init_method)
        init_weights(self.b4_cls.modules(),init_method)
        init_weights(self.b3_cls.modules(),init_method)
        init_weights(self.b2_cls.modules(),init_method)
        init_weights(self.b1_cls.modules(),init_method)

    def forward(self,x):
        feat=self.basenet(x)
        # bg fg txt
        b4_mask = self.b4_mask(feat.b4)
        b3_mask = self.b3_mask(feat.b3)
        # txt,edge
        b2_mask = self.b2_mask(feat.b2)
        b1_mask = self.b1_mask(feat.b1)
        b4_mask = F.interpolate(b4_mask, size=b1_mask.size()[2:], mode='bilinear', align_corners=False)
        b3_mask = F.interpolate(b3_mask, size=b1_mask.size()[2:], mode='bilinear', align_corners=False)
        b2_mask = F.interpolate(b2_mask, size=b1_mask.size()[2:], mode='bilinear', align_corners=False)
        edg_mask = F.elu(F.elu(F.elu(b1_mask[:,1])+b2_mask[:,1])-b3_mask[:,0])-b4_mask[:,0]
        txt_mask = F.elu(F.elu(F.elu(F.elu(b1_mask[:,0])+b2_mask[:,0])+b3_mask[:,2]-b3_mask[:,0])+b4_mask[:,2]-b4_mask[:,0])
        region_mask = F.elu(F.elu(b3_mask[:,1]-b3_mask[:,0])+b4_mask[:,1]-b4_mask[:,0])
        mask = torch.stack((txt_mask,edg_mask,region_mask),1)
        b4_cls = self.b4_cls(feat.b4)
        b3_cls = self.b3_cls(feat.b3)
        b2_cls = self.b2_cls(feat.b2)
        b1_cls = self.b1_cls(feat.b1)
        f_cls = b4_cls
        f_cls = b3_cls+F.interpolate(f_cls, size=b3_cls.size()[2:], mode='bilinear', align_corners=False)
        f_cls = b2_cls+F.interpolate(f_cls, size=b2_cls.size()[2:], mode='bilinear', align_corners=False)
        f_cls = b1_cls+F.interpolate(f_cls, size=b1_cls.size()[2:], mode='bilinear', align_corners=False)
        mask = torch.cat((mask,f_cls),1)

        featout = self.out_tuple(b1_mask, b2_mask, b3_mask, b4_mask, feat.b1,feat.b2,feat.b3,feat.b4)
        return mask,featout

class PIX_Unet(nn.Module):
    def __init__(self,basenet:str ='mobile', min_upc_ch:int =128,
        padding:bool=True, init_method:str='xavier_uniform',
        **args):
        """
        Basenet Args:
            basenet: basenet name
            min_upc_ch: minimum up-convolution channel num
            init_method: weight initial method
        Task Args:
            mask_ch: int, enable mask prediction if given
                min_mask_ch: optional, minimum ch in conv. group

            cls_ch: int, enable classification prediction if given
                min_cls_ch: optional, minimum ch in conv. group

            box_ch: int, enable box prediction if given
                hitmap_ch: default 2, hitmap ch for box
                min_box_ch: optional, minimum ch in conv. group

            the final prediction will be [mask_ch,cls_ch,hitmap_ch,box_ch]

        """
        super(PIX_Unet, self).__init__()
        basenet = basenet.lower()
        if('vgg' in basenet):
            self.basenet = VGGUnet(basenet=basenet,min_upc_ch=min_upc_ch,padding=padding,init_method=init_method,**args)
        elif('resnet' in basenet):
            self.basenet = ResnetUnet(basenet=basenet,min_upc_ch=min_upc_ch,init_method=init_method,**args)
        else:
            self.basenet = MobUNet(basenet=basenet,min_upc_ch=min_upc_ch,init_method=init_method,**args)
        upch = self.basenet.out_channels

        self.b_mask_task = 'mask_ch' in args
        self.b_cls_task = 'cls_ch' in args
        self.b_box_task = 'box_ch' in args

        if(self.b_mask_task):
            min_map_ch = args['min_mask_ch'] if('min_mask_ch' in args)else upch//8
            mask_ch = args['mask_ch']
            self.mask = nn.Sequential(
                double_conv(upch,max(upch//2,min_map_ch),max(upch//4,min_map_ch)),
                double_conv(max(upch//4,min_map_ch),max(upch//8,min_map_ch),mask_ch),
                )
            init_weights(self.mask.modules(),init_method)
            
        if(self.b_cls_task):
            self.multi_level = bool(args['multi_level']) if('multi_level' in args)else False
            cls_ch = args['cls_ch']
            min_cls_ch = args['min_cls_ch'] if('min_cls_ch' in args)else 0
            if(self.multi_level):
                b0ch = get_final_ch(self.basenet.basenet.b0)
                b1ch = get_final_ch(self.basenet.basenet.b1)
                b2ch = get_final_ch(self.basenet.basenet.b2)
                b3ch = get_final_ch(self.basenet.basenet.b3)
                b4ch = get_final_ch(self.basenet.basenet.b4)
                b4ch = get_final_ch(self.basenet.basenet.b4)

                self.b4_cls = nn.Sequential(
                    double_conv(b4ch,max(b4ch//2,min_cls_ch),max(b4ch//4,min_cls_ch)),
                    double_conv(max(b4ch//4,min_cls_ch),max(b4ch//8,min_cls_ch),cls_ch),
                    )
                self.b3_cls = nn.Sequential(
                    double_conv(b3ch,max(b3ch//2,min_cls_ch),max(b3ch//4,min_cls_ch)),
                    double_conv(max(b3ch//4,min_cls_ch),max(b3ch//8,min_cls_ch),cls_ch),
                    )
                self.b2_cls = nn.Sequential(
                    double_conv(b2ch,max(b2ch//2,min_cls_ch),max(b2ch//4,min_cls_ch)),
                    double_conv(max(b2ch//4,min_cls_ch),max(b2ch//8,min_cls_ch),cls_ch),
                    )
                self.b1_cls = nn.Sequential(
                    double_conv(b1ch,max(b1ch//2,min_cls_ch),max(b1ch//4,min_cls_ch)),
                    double_conv(max(b1ch//4,min_cls_ch),max(b1ch//8,min_cls_ch),cls_ch),
                    )
                init_weights(self.b4_cls.modules(),init_method)
                init_weights(self.b3_cls.modules(),init_method)
                init_weights(self.b2_cls.modules(),init_method)
                init_weights(self.b1_cls.modules(),init_method)
            else:
                upch = self.basenet.out_channels
                self.cls = nn.Sequential(
                    double_conv(upch,max(upch//2,min_cls_ch),max(upch//4,min_cls_ch)),
                    double_conv(max(upch//4,min_cls_ch),max(upch//8,min_cls_ch),cls_ch),
                    )
                init_weights(self.cls.modules(),init_method)

        if(self.b_box_task):
            min_box_ch = args['min_box_ch'] if('min_box_ch' in args)else 0
            hitmap_ch = args['hitmap_ch'] if('hitmap_cn' in args)else 2
            box_ch = args['box_ch']
            self.box = nn.Sequential(
                double_conv(upch,max(upch//2,min_box_ch),max(upch//4,min_box_ch)),
                double_conv(max(upch//4,min_box_ch),max(upch//8,min_box_ch),box_ch),
                )
            self.hitmap = nn.Sequential(
                double_conv(upch,max(upch//2,min_box_ch),max(upch//4,min_box_ch)),
                double_conv(max(upch//4,min_box_ch),max(upch//8,min_box_ch),hitmap_ch),
                )
            init_weights(self.box.modules(),init_method)

        assert (self.b_mask_task or self.b_cls_task or self.b_box_task), "At least need one task"

    def forward(self,x):
        feat=self.basenet(x)
        preds = []
        if(self.b_mask_task):
            preds.append(self.mask(feat.upb0))
        if(self.b_cls_task):
            if(self.multi_level):
                b4_cls = self.b4_cls(feat.b4)
                b3_cls = self.b3_cls(feat.b3)
                b2_cls = self.b2_cls(feat.b2)
                b1_cls = self.b1_cls(feat.b1)
                f_cls = b4_cls
                f_cls = b3_cls+F.interpolate(f_cls, size=b3_cls.size()[2:], mode='bilinear', align_corners=False)
                f_cls = b2_cls+F.interpolate(f_cls, size=b2_cls.size()[2:], mode='bilinear', align_corners=False)
                f_cls = b1_cls+F.interpolate(f_cls, size=b1_cls.size()[2:], mode='bilinear', align_corners=False)
                if(f_cls.shape[2:]!=pred.shape[2:]):
                    f_cls = F.interpolate(f_cls, size=pred.size()[2:], mode='bilinear', align_corners=False)
            else:
                f_cls = self.cls(feat.upb0)
            preds.append(f_cls)
        if(self.b_box_task):
            preds.append(self.hitmap(feat.upb0))
            preds.append(self.box(feat.upb0))

        return torch.cat(preds,1),feat

class PIXCNN(nn.Module):
    def __init__(self, mask_ch:int = 2, basenet:str ='mobile', min_upc_ch:int =128,
        init_method:str='xavier_uniform',
        **args):
        """
        Basenet Args:
            mask_ch: int, enable mask prediction if given
            basenet: basenet name
            min_upc_ch: minimum up-convolution channel num
            init_method: weight initial method
            min_mask_ch: optional, minimum ch in conv. group
        """
        super(PIXCNN, self).__init__()
        basenet = basenet.lower()
        if('vgg' in basenet):
            # set padding=True because the input feature image size of LSTM is fixed
            self.basenet = VGGUnet(basenet=basenet,min_upc_ch=min_upc_ch,padding=True,init_method=init_method,**args)
        elif('resnet' in basenet):
            self.basenet = ResnetUnet(basenet=basenet,min_upc_ch=min_upc_ch,init_method=init_method,**args)
        else:
            self.basenet = MobUNet(basenet=basenet,min_upc_ch=min_upc_ch,init_method=init_method,**args)
        upch = self.basenet.out_channels

        min_map_ch = args['min_mask_ch'] if('min_mask_ch' in args)else upch//8

        self.mask_pre_filter = double_conv(
            upch,max(upch//2,min_map_ch),max(upch//4,min_map_ch),
            kernel_size=(3,3))

        self.final_predict_ch = max(upch//4,min_map_ch)
        self.mask_predictor = double_conv(
            self.final_predict_ch,max(self.final_predict_ch//2,min_map_ch),mask_ch,
            kernel_size=(3,3))
            
        init_weights(self.mask_pre_filter.modules(),init_method)
        init_weights(self.mask_predictor.modules(),init_method)

    def forward(self,x):
        feat=self.basenet(x)
        f_filt = self.mask_pre_filter(feat.upb0)
        pred = self.mask_predictor(f_filt)

        return pred,feat

class PIXLSTM(nn.Module):
    def __init__(self, mask_ch:int = 2, basenet:str ='mobile', min_upc_ch:int =128,
        init_method:str='xavier_uniform',
        **args):
        """
        Basenet Args:
            mask_ch: int, enable mask prediction if given
            basenet: basenet name
            min_upc_ch: minimum up-convolution channel num
            init_method: weight initial method
            min_mask_ch: optional, minimum ch in conv. group
        """
        super(PIXLSTM, self).__init__()
        basenet = basenet.lower()
        if('vgg' in basenet):
            # set padding=True because the input feature image size of LSTM is fixed
            self.basenet = VGGUnet(basenet=basenet,min_upc_ch=min_upc_ch,padding=True,init_method=init_method,**args)
        elif('resnet' in basenet):
            self.basenet = ResnetUnet(basenet=basenet,min_upc_ch=min_upc_ch,init_method=init_method,**args)
        else:
            self.basenet = MobUNet(basenet=basenet,min_upc_ch=min_upc_ch,init_method=init_method,**args)
        upch = self.basenet.out_channels

        min_map_ch = args['min_mask_ch'] if('min_mask_ch' in args)else upch//8

        self.mask_pre_filter = double_conv(
            upch,max(upch//2,min_map_ch),max(upch//4,min_map_ch),
            kernel_size=(3,3))

        self.final_predict_ch = max(upch//4,min_map_ch)
        self.mask_predictor = double_conv(
            self.final_predict_ch,max(self.final_predict_ch//2,min_map_ch),mask_ch,
            kernel_size=(3,3))
            
        init_weights(self.mask_pre_filter.modules(),init_method)
        init_weights(self.mask_predictor.modules(),init_method)
            
        self.lstm = BottleneckLSTMCell(input_channels=self.final_predict_ch,
            hidden_channels=self.final_predict_ch)
        self.lstmh,self.lstmc = None,None

    def init_state(self,shape=(320,320),batch_size=1):
        _,d = next(iter(self.state_dict().items()))
        self.lstmh = torch.zeros((batch_size,self.final_predict_ch,shape[0],shape[1]),dtype=d.dtype).to(d.device)
        self.lstmc = torch.zeros((batch_size,self.final_predict_ch,shape[0],shape[1]),dtype=d.dtype).to(d.device)

        if(self.lstm.Wci is None):
            self.lstm.Wci = nn.Parameter(torch.rand(1, self.final_predict_ch, shape[0], shape[1],dtype=d.dtype),requires_grad=True).to(d.device)
            self.lstm.Wcf = nn.Parameter(torch.rand(1, self.final_predict_ch, shape[0], shape[1],dtype=d.dtype),requires_grad=True).to(d.device)
            self.lstm.Wco = nn.Parameter(torch.rand(1, self.final_predict_ch, shape[0], shape[1],dtype=d.dtype),requires_grad=True).to(d.device)

    def forward(self,x):
        feat=self.basenet(x)
        f_filt = self.mask_pre_filter(feat.upb0)
        self.lstmh,self.lstmc = self.lstm(f_filt,self.lstmh,self.lstmc)
        pred = self.mask_predictor(self.lstmh)

        return pred,feat

class PIXLSTM_Residual(nn.Module):
    def __init__(self, mask_ch:int = 2, basenet:str ='mobile', min_upc_ch:int =128,
        init_method:str='xavier_uniform',
        **args):
        """
        Basenet Args:
            mask_ch: int, enable mask prediction if given
            basenet: basenet name
            min_upc_ch: minimum up-convolution channel num
            init_method: weight initial method
            min_mask_ch: optional, minimum ch in conv. group
        """
        super(PIXLSTM_Residual, self).__init__()
        basenet = basenet.lower()
        if('vgg' in basenet):
            # set padding=True because the input feature image size of LSTM is fixed
            self.basenet = VGGUnet(basenet=basenet,min_upc_ch=min_upc_ch,padding=True,init_method=init_method,**args)
        elif('resnet' in basenet):
            self.basenet = ResnetUnet(basenet=basenet,min_upc_ch=min_upc_ch,init_method=init_method,**args)
        else:
            self.basenet = MobUNet(basenet=basenet,min_upc_ch=min_upc_ch,init_method=init_method,**args)
        upch = self.basenet.out_channels

        min_map_ch = args['min_mask_ch'] if('min_mask_ch' in args)else upch//8

        self.mask_pre_filter = double_conv(
            upch,max(upch//2,min_map_ch),max(upch//4,min_map_ch),
            kernel_size=(3,3))

        self.final_predict_ch = max(upch//4,min_map_ch)
        self.mask_predictor = double_conv(
            self.final_predict_ch,max(self.final_predict_ch//2,min_map_ch),mask_ch,
            kernel_size=(3,3))
            
        init_weights(self.mask_pre_filter.modules(),init_method)
        init_weights(self.mask_predictor.modules(),init_method)
            
        self.lstm = BottleneckLSTMCell(input_channels=self.final_predict_ch,
            hidden_channels=self.final_predict_ch)
        self.residual_bn = nn.BatchNorm2d(self.final_predict_ch)
        self.lstmh,self.lstmc = None,None

    def init_state(self,shape=(320,320),batch_size=1):
        _,d = next(iter(self.state_dict().items()))
        self.lstmh = torch.zeros((batch_size,self.final_predict_ch,shape[0],shape[1]),dtype=d.dtype).to(d.device)
        self.lstmc = torch.zeros((batch_size,self.final_predict_ch,shape[0],shape[1]),dtype=d.dtype).to(d.device)

        if(self.lstm.Wci is None):
            self.lstm.Wci = nn.Parameter(torch.rand(1, self.final_predict_ch, shape[0], shape[1],dtype=d.dtype),requires_grad=True).to(d.device)
            self.lstm.Wcf = nn.Parameter(torch.rand(1, self.final_predict_ch, shape[0], shape[1],dtype=d.dtype),requires_grad=True).to(d.device)
            self.lstm.Wco = nn.Parameter(torch.rand(1, self.final_predict_ch, shape[0], shape[1],dtype=d.dtype),requires_grad=True).to(d.device)

    def forward(self,x):
        feat=self.basenet(x)
        f_filt = self.mask_pre_filter(feat.upb0)
        self.lstmh,self.lstmc = self.lstm(f_filt,self.lstmh,self.lstmc)
        f_res = self.residual_bn(f_filt+self.lstmh)
        pred = self.mask_predictor(f_res)

        return pred,feat

class VGG_PXMASK(nn.Module):
    def __init__(self,padding:bool=True,
        init_method:str='xavier_uniform',**args):
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

        init_weights(self.b4_mask.modules(),init_method)
        init_weights(self.b3_mask.modules(),init_method)
        init_weights(self.b2_mask.modules(),init_method)
        init_weights(self.b1_mask.modules(),init_method)
        init_weights(self.b0_mask.modules(),init_method)

    def forward(self,x):
        b0, b1, b2, b3, b4 = self.basenet(x)

        b4_mask = self.b4_mask(b4)
        b3_mask = self.b3_mask(b3)
        b2_mask = self.b2_mask(b2)
        b1_mask = self.b1_mask(b1)
        b0_mask = self.b0_mask(b0)
        mask = b4_mask
        mask = b3_mask+F.interpolate(mask, size=b3_mask.shape[2:], mode='bilinear', align_corners=False)
        mask = b2_mask+F.interpolate(mask, size=b2_mask.shape[2:], mode='bilinear', align_corners=False)
        mask = b1_mask+F.interpolate(mask, size=b1_mask.shape[2:], mode='bilinear', align_corners=False)
        mask = b0_mask+F.interpolate(mask, size=b0_mask.shape[2:], mode='bilinear', align_corners=False)
        feat = namedtuple("pixelmap", ['b0_mask','b1_mask', 'b2_mask', 'b3_mask', 'b4_mask', 'b1','b2','b3','b4'])
        featout = feat(b0_mask, b1_mask, b2_mask, b3_mask, b4_mask, b1,b2,b3,b4)
        return mask,featout

class VGGUnet_PXMASK(nn.Module):
    def __init__(self,padding:bool=True,
        init_method:str='xavier_uniform',**args):
        super(VGGUnet_PXMASK, self).__init__()
        self.basenet = VGGUnet(padding=padding,init_method=init_method,**args)

        self.final_mask = nn.Sequential(
            double_conv(self.basenet.out_channels,128,64,padding=padding),
            double_conv(64,32,3,padding=padding),
            )
        self.init_weights(self.final_mask.modules(),init_method)

    def forward(self,x):
        vgg_feat = self.basenet(x)

        mask = self.final_mask(vgg_feat.upb0)
        
        return mask,unet_feat

class VGG_PUR_CLS(nn.Module):
    def __init__(self,padding:bool=True,
        init_method:str='xavier_uniform',**args):
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
        init_weights(self.b4_cls.modules(),init_method)
        init_weights(self.b3_cls.modules(),init_method)
        init_weights(self.b2_cls.modules(),init_method)
        init_weights(self.b1_cls.modules(),init_method)
        init_weights(self.b0_cls.modules(),init_method)

    def forward(self,x):
        vgg_feat = self.basenet(x)
        b0, b1, b2, b3, b4 = vgg_feat
        b4_clsp = self.b4_cls(b4)
        b3_clsp = self.b3_cls(b3)
        b2_clsp = self.b2_cls(b2)
        b1_clsp = self.b1_cls(b1)
        b0_clsp = self.b0_cls(b0)
        final_cls = b4_clsp
        final_cls = b3_clsp+F.interpolate(final_cls, size=b3_clsp.size()[2:], mode='bilinear')
        final_cls = b2_clsp+F.interpolate(final_cls, size=b2_clsp.size()[2:], mode='bilinear')
        final_cls = b1_clsp+F.interpolate(final_cls, size=b1_clsp.size()[2:], mode='bilinear')
        final_cls = b0_clsp+F.interpolate(final_cls, size=b0_clsp.size()[2:], mode='bilinear')
        return final_cls,vgg_feat