from __future__ import print_function, division
import os
import numpy as np
import cv2
from lib.utils.img_hlp import np_apply_matrix_to_pts
from . import base

class MSRA(base.BaseDataset):
    """
    Reference:
        @INPROCEEDINGS{6247787,
            author={C. {Yao} and X. {Bai} and W. {Liu} and Y. {Ma} and Z. {Tu}},
            booktitle={2012 IEEE Conference on Computer Vision and Pattern Recognition}, 
            title={Detecting texts of arbitrary orientations in natural images}, 
            year={2012},
            volume={},
            number={},
            pages={1083-1090},
            doi={10.1109/CVPR.2012.6247787}}
    The text format is:
        index, default label, x,y, w,h, rotation angle

    """
    def __init__(self, img_dir, gt_txt_dir, out_box_format='polyxy',
        **params):
        in_box_format = 'polyxy'
        gt_txt_name_lambda = lambda x: "%s.gt"%x
        gt_mask_name_lambda = None

        super(MSRA,self).__init__(img_dir=img_dir, gt_mask_dir=None, gt_txt_dir=gt_txt_dir, in_box_format=in_box_format,
        gt_txt_name_lambda=gt_txt_name_lambda, out_box_format=out_box_format,
        **params)

    def read_boxs(self,fname:str):
        """
        index, default label, x,y, w,h, rotation angle
        """
        f = open(fname,'r')
        lines = f.readlines()
        f.close()
        boxs = []

        for line in lines:
            datas = [float(o) for o in line.split(' ')]
            index,label,x,y, w,h,theta = datas
            bx = np.array([(x,y,1),(x+w,y,1),(x+w,y+h,1),(x,y+h,1)])
            cent = (x+w/2,y+h/2)
            M = cv2.getRotationMatrix2D(cent, -theta/np.pi*180, 1.0)
            bx = np.dot(M,np.moveaxis(bx,-1,0))
            bx = np.moveaxis(bx,0,-1)
            boxs.append(bx)

        return np.array(boxs),None

    # def post_process(self,sample,fname):
    #     return sample