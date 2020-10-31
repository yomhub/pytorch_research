from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from . import base

def read_boxs_poly(fname:str):
    # read polygon box and return box,text
    f = open(fname, "r", encoding='utf-8-sig')
    lines = f.readlines()
    f.close()
    boxes = []
    text = []
    for line in lines:
        o = line.split(',')
        tmp = ''
        for ch in o[8:]:
            if(ch):
                tmp+=ch.strip('\n')
        if(tmp=='###'):
            continue
        boxes.append(np.array([int(d) for d in o[:8]]).reshape(-1,2))
        text.append(tmp)

    return np.array(boxes),text

def read_boxs_2p(fname:str):
    # read "x y x y text" box and return polygon box and text
    f = open(fname, "r", encoding='utf-8-sig')
    lines = f.readlines()
    f.close()
    boxes = []
    text = []
    for line in lines:
        o = line.split(' ')
        tmp=o[-1].strip().strip('"')
        if(tmp=='###'):
            continue
        text.append(tmp)
        x0,y0,x1,y1 = int(o[0]),int(o[1]),int(o[2]),int(o[3])
        boxes.append([x0,y0,x1,y0,x1,y1,x0,y1])

    return np.array(boxes).reshape(-1,4,2),text

class ICDAR19(base.BaseDataset):
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
        gt_txt_name_lambda = lambda x: "%s.txt"%x

        super(ICDAR19,self).__init__(img_dir=img_dir, gt_txt_dir=gt_txt_dir, 
        in_box_format=in_box_format,gt_txt_name_lambda=gt_txt_name_lambda, 
        out_box_format=out_box_format,
        **params)

    def read_boxs(self,fname:str):
        return read_boxs_poly(fname)

class ICDAR15(base.BaseDataset):
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

        super(ICDAR15,self).__init__(img_dir=img_dir, gt_txt_dir=gt_txt_dir, 
        in_box_format=in_box_format,gt_txt_name_lambda=gt_txt_name_lambda, 
        out_box_format=out_box_format,
        **params)

    def read_boxs(self,fname:str):
        return read_boxs_poly(fname)

class ICDAR13(base.BaseDataset):
    """
    ICDAR dataset
    The text format is:
        One line represent one box.
        2 points, text
        x0 y0 x1 y1 text
    Args: 
        img_dirs: image dir, string or list
        gt_dirs: image dir, string or list or none
    """

    def __init__(self, img_dir, gt_txt_dir,
        out_box_format = 'polyxy',
        **params):
        in_box_format = 'polyxy'
        gt_txt_name_lambda = lambda x: "gt_%s.txt"%x

        super(ICDAR13,self).__init__(img_dir=img_dir, gt_txt_dir=gt_txt_dir, 
        in_box_format=in_box_format,gt_txt_name_lambda=gt_txt_name_lambda, 
        out_box_format=out_box_format,
        **params)

    def read_boxs(self,fname:str):
        return read_boxs_2p(fname)