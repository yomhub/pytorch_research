import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.model.craft import CRAFT
from lib.utils.img_hlp import cv_get_box_from_mask
from lib.utils.net_hlp import init_weights,Swish_act,double_conv

class SiameseNet(nn.Module):
    """ The basic siamese network joining network, that takes the outputs of
    two embedding branches and joins them applying a correlation operation.
    Should always be used with tensors of the form [B x C x H x W], i.e.
    you must always include the batch dimension.
    """

    def __init__(self, embedding_net, upscale=False, corr_map_size=33, stride=4):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.match_batchnorm = nn.BatchNorm2d(1)

        self.upscale = upscale
        # TODO calculate automatically the final size and stride from the
        # parameters of the branch
        self.corr_map_size = corr_map_size
        self.stride = stride
        # Calculates the upscale size based on the correlation map size and
        # the total stride of the network, so as to align the corners of the
        # original and the upscaled one, which also aligns the centers.
        self.upsc_size = (self.corr_map_size-1)*self.stride + 1
        # The upscale_factor is the correspondence between a movement in the output
        # feature map and the input images. So if a network has a total stride of 4
        # and no deconvolutional or upscaling layers, a single pixel displacement
        # in the output corresponds to a 4 pixels displacement in the input
        # image. The easiest way to compensate this effect is to do a bilinear
        # or bicubic upscaling.
        if upscale:
            self.upscale_factor = 1
        else:
            self.upscale_factor = self.stride

    def forward(self, x):
        pred, feature = self.embedding_net(x)

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): The reference patch of dimensions [B, C, H, W].
                Usually the shape is [8, 3, 127, 127].
            x2 (torch.Tensor): The search region image of dimensions
                [B, C, H', W']. Usually the shape is [8, 3, 255, 255].
        Returns:
            match_map (torch.Tensor): The score map for the pair. For the usual
                input shapes, the output shape is [8, 1, 33, 33].
        """
        embedding_reference = self.embedding_net(x1)
        embedding_search = self.embedding_net(x2)
        match_map = self.match_corr(embedding_reference, embedding_search)
        return match_map

    def get_embedding(self, x):
        return self.embedding_net(x)

    def match_corr(self, embed_ref, embed_srch):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].
        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.
        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        b, c, h, w = embed_srch.shape
        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w),
                             embed_ref, groups=b)
        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)
        if self.upscale:
            match_map = F.interpolate(match_map, self.upsc_size, mode='bilinear',
                                      align_corners=False)

        return match_map

class SiameseCRAFT(nn.Module):
    def __init__(self, base_net, feature_chs, lock_basenet:bool=True):
        super(SiameseCRAFT, self).__init__()
        self.base_net = base_net
        if(lock_basenet):
            for i in self.base_net.parameters():
                i.requires_grad=False
        self.match_batchnorm = nn.BatchNorm2d(feature_chs)
        self.search_norm = nn.BatchNorm2d(feature_chs)
        self.obj_norm = nn.BatchNorm2d(feature_chs)
        self.map_conv = nn.Sequential(
            nn.Conv2d(feature_chs, feature_chs//2, kernel_size=3, padding=0), nn.ReLU(),#Swish_act()
            nn.Conv2d(feature_chs//2, feature_chs//4, kernel_size=3, padding=0), nn.ReLU(),#nn.ReLU
            nn.Conv2d(feature_chs//4, 1, kernel_size=1),
        )
        # self.final_act_fun = lambda x: torch.exp(-x*x/1.62)
        self.final_act_fun = lambda x: x
        
        # init_weights(self.map_conv)
    def forward(self, x):
        return self.base_net(x)
    def match(self,obj,search):
        search = self.search_norm(search)
        obj = self.obj_norm(obj)
        match_map = conv2d_dw_group(search,obj)
        match_map = self.match_batchnorm(match_map)
        score = self.map_conv(match_map)
        score = self.final_act_fun(score)
        return score,match_map

class SiamesePXT(nn.Module):
    def __init__(self, base_net, feature_chs, lock_basenet:bool=True):
        super(SiamesePXT, self).__init__()
        self.base_net = base_net
        if(lock_basenet):
            for i in self.base_net.parameters():
                i.requires_grad=False
        self.match_batchnorm = nn.BatchNorm2d(feature_chs)
        self.search_norm = nn.BatchNorm2d(feature_chs)
        self.obj_norm = nn.BatchNorm2d(feature_chs)
        self.map_conv = nn.Sequential(
            nn.Conv2d(feature_chs, feature_chs//2, kernel_size=3, padding=0), nn.ReLU(),#Swish_act()
            nn.Conv2d(feature_chs//2, feature_chs//4, kernel_size=3, padding=0), nn.ReLU(),#nn.ReLU
            nn.Conv2d(feature_chs//4, 1, kernel_size=1),
        )

        # init_weights(self.map_conv)
    def forward(self, x):
        return self.base_net(x)
    def match(self,obj,search):
        obj = self.obj_norm(obj)
        match_map = conv2d_dw_group(search,obj)
        score = self.map_conv(match_map)
        return score,match_map

def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
