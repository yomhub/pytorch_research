from __future__ import print_function, division
import os
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from . import base

class Total(base.BaseDataset):
    """
    Total dataset, reference below:
        @article{CK2019,
            author    = {Chee Kheng Ch’ng and
                        Chee Seng Chan and
                        Chenglin Liu},
            title     = {Total-Text: Towards Orientation Robustness in Scene Text Detection},
            journal   = {International Journal on Document Analysis and Recognition (IJDAR)},
            volume    = {23},
            pages     = {31-52},
            year      = {2020},
            doi       = {10.1007/s10032-019-00334-z},
            }
    The text format is:
        One line represent one box.
        Column 1-2 = X-coordinate
        Column 3-4 = Y-coordinate
        Column 5 = Text
        Column 6 = Orientation (c=curve; h=horizontal; m=multi-oriented; #=dont care)
    Args:
        'box_format': string in ['yxyx','xyxy','xywh','cxywh','polyxy']
            'yxyx': box_cord = [y1,x1,y2,x2]
            'xyxy': box_cord = [x1,y1,x2,y2]
            'xywh': box_cord = [x,y,w,h]
            'cxywh': box_cord = [cx,cy,w,h]
            'polyxy': box_cord = [x,y,x,y,x,y,x,y]
        normalized: True to normalize coordinate
    """
    def __init__(self, img_dir, gt_mask_dir, gt_txt_dir, out_box_format='polyxy',include_bg:bool=False,
        **params):
        in_box_format = 'polyxy'
        gt_txt_name_lambda = lambda x: "poly_gt_%s.txt"%x
        gt_mask_name_lambda = None
        self.include_bg = include_bg
        super(Total,self).__init__(img_dir=img_dir, gt_mask_dir=gt_mask_dir, gt_txt_dir=gt_txt_dir, in_box_format=in_box_format,
        gt_mask_name_lambda=gt_mask_name_lambda, gt_txt_name_lambda=gt_txt_name_lambda, out_box_format=out_box_format,
        **params)

    def read_boxs(self,fname:str):
        """
        x: [[ x1 x2 ...]], y: [[y1 y2 ...]], ornt: [u't'], transcriptions: [u'texts']
        """
        f = open(fname,'r')
        lines = f.readlines()
        boxs = []
        txts = []

        i = 0

        while i < len(lines):
            line = lines[i].strip()
            while not line.endswith(']'):
                i = i + 1
                line = line + ' ' + lines[i].strip()
            i += 1
            parts = line.split(',')
            ort = parts[2].split()[-1]

            xs = [int(o) for o in parts[0].split('[[')[-1].split(']]')[0].split()]
            ys = [int(o) for o in parts[1].split('[[')[-1].split(']]')[0].split()]
            if(not(len(xs)==len(ys) and len(xs)>=3)):
                continue
            # if(len(ort)>4 and ort[3]=='#'):
            #     # skip Orientation = #
            #     continue
            txt = '#' if('#' in ort)else parts[3].split()[-1][3:-2]
            if(not self.include_bg and txt=='#'):
                continue
            if('poly' in self.out_box_format):
                boxs.append(np.asarray([(xi,yi) for xi,yi in zip(xs,ys)]))
            else:
                boxs.append([int(1), min(ys), min(xs),max(ys), max(xs)])
            txts.append(txt)
        return np.array(boxs),txts

    # def post_process(self,sample,fname):
    #     return sample