from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from . import base

class ICDAR(base.BaseDataset):
    """
    ICDAR dataset
    The text format is:
        One line represent one box.
        4 points, language, text
        x0,y0,x1,y1,x2,y2,x3,y3,language,text
    Args: 
        img_dirs: image dir, string or list
        gt_dirs: image dir, string or list or none
    """

    def __init__(self, img_dir, gt_txt_dir,
        out_box_format = 'polyxy',
        **params):
        in_box_format = 'polyxy'
        gt_txt_name_lambda = lambda x: "gt_%s.txt"%x

        super(ICDAR,self).__init__(img_dir=img_dir, gt_txt_dir=gt_txt_dir, 
        in_box_format=in_box_format,gt_txt_name_lambda=gt_txt_name_lambda, 
        out_box_format=out_box_format,
        **params)

    def read_boxs(self,fname:str):
        f = open(fname, "r", encoding='utf-8-sig')
        lines = f.readlines()
        f.close()
        boxes = []
        text = []
        for line in lines:
            o = line.split(',')
            boxes.append([int(d) for d in o[:8]])
            tmp = ''
            for ch in o[8:]:
                if(ch):
                    tmp+=ch.strip('\n')
            text.append(tmp)

        return np.array(boxes,dtype=np.uint8),text
