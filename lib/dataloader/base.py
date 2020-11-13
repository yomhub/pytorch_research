from __future__ import print_function, division
import os
from collections import Iterable
from skimage import io, transform
import numpy as np
import torch
import torchvision
import cv2
from torch.utils.data import Dataset
from torchvision import transforms, utils
from lib.utils.img_hlp import np_box_transfrom,np_box_nor,np_box_resize,np_img_normalize

def default_collate_fn(batch):
    # batch: list of dict
    ret = {}
    for key,value in batch[0].items():
        if(key.lower() in ['box','text','name'] or 'list' in key.lower()):
            ret[key]=[d[key] for d in batch]
        elif(key.lower() in ['box_format'] or 'sig' in key.lower()):
            ret[key]=value[0] if(isinstance(value,list))else value
        else:
            ret[key]=torch.stack([torch.from_numpy(d[key].copy())if(isinstance(d[key],np.ndarray))else d[key] for d in batch],0)
    return ret

def default_x_input_function(sample,th_device): 
    x = sample['image'] if(isinstance(sample,dict))else sample
    return x.permute((0,3,1,2)).float().to(th_device)
    
class BaseDataset(Dataset):
    """
    Args:
        img_dir: str or list, image folder
        'box_format': string in ['yxyx','xyxy','xywh','cxywh']
            'yxyx': box_cord = [y1,x1,y2,x2]
            'xyxy': box_cord = [x1,y1,x2,y2]
            'xywh': box_cord = [x,y,w,h]
            'cxywh': box_cord = [cx,cy,w,h]
        normalized: True to normalize coordinate
        max_image_size/image_size: (y,x),int or float for both (y,x)
    Outs:
        {
            'image': (h,w,1 or 3) np array.
            'video': cv <VideoCapture object>

            If have gt_txt_dir,
            'box': (k,4 or 5) np array, where 4 or 5 is (1 or 0)+(box_cord,4)
            'box_format': string in ['yxyx','xyxy','xywh','cxywh']
                'yxyx': box_cord = [y1,x1,y2,x2]
                'xyxy': box_cord = [x1,y1,x2,y2]
                'xywh': box_cord = [x,y,w,h]
                'cxywh': box_cord = [cx,cy,w,h]
            'text': list of texts

            If have gt_mask_dir,
            'gtmask': (h,w,1) np array.
        }
    """

    def __init__(self, img_dir, gt_mask_dir=None, gt_txt_dir=None, in_box_format:str=None,
        gt_mask_name_lambda=None, gt_txt_name_lambda=None, 
        out_box_format:str='cxywh', normalized=False, transform=None,
        image_size=None, max_image_size=None, img_only = False):

        self.in_box_format = in_box_format.lower() if(in_box_format!=None)else None
        self.out_box_format = out_box_format.lower()

        self.normalize = bool(normalized)
        self.imgdir = img_dir if(type(img_dir) in [list,tuple])else [img_dir]
        self.gt_mask_dir = gt_mask_dir
        self.gt_mask_name_lambda = gt_mask_name_lambda
        self.gt_txt_dir = gt_txt_dir
        self.gt_txt_name_lambda = gt_txt_name_lambda
        self.type_list = ['jpg','png','bmp'] if(img_only)else ['jpg','png','bmp','mp4','avi']
        self.img_names = [os.path.join(path,o) for fld in self.imgdir for path,dir_list,file_list in os.walk(fld) for o in file_list if o.lower().split('.')[-1] in self.type_list]

        self.transform=transform
        if(isinstance(image_size,type(None))):
            self.image_size = None
        elif(not isinstance(image_size,Iterable)):
            self.image_size = (image_size,image_size)
        else:
            self.image_size = image_size

        if(isinstance(max_image_size,type(None))):
            self.max_image_size = None
        elif(not isinstance(max_image_size,Iterable)):
            self.max_image_size = (max_image_size,max_image_size)
        else:
            self.max_image_size = max_image_size

        self.default_collate_fn = default_collate_fn
        self.x_input_function = default_x_input_function

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}
        if(self.img_names[idx].split('.')[-1].lower() in self.type_list):
            try:
                img = io.imread(self.img_names[idx])
                if(len(img.shape)==2):
                    img = np.expand_dims(img,-1)
                    img = np.broadcast_to(img,(img.shape[0],img.shape[1],3))
                img = img[:,:,:3]
            except Exception as e:
                raise RuntimeError("Err when reading {}: {}".format(self.img_names[idx],e))

            org_shape = img.shape[0:2]
            # if(not isinstance(self.image_size,type(None)) and img.shape[0:2]!=self.image_size):
            if(not isinstance(self.image_size,type(None))):
                img = transform.resize(img,self.image_size,preserve_range=True)
            elif(not isinstance(self.max_image_size,type(None)) and (org_shape[0]>self.max_image_size[0] or org_shape[1]>self.max_image_size[1])):
                img = transform.resize(img,(min(org_shape[0],self.max_image_size[0]),min(org_shape[1],self.max_image_size[1])),preserve_range=True)

            sample = {'image': img}
        elif(self.img_names[idx].split('.')[-1].lower() in self.vdo_type):
            vfile = cv2.VideoCapture(self.img_names[idx])
            sample = {'video': vfile}

        img_nm = os.path.basename(self.img_names[idx]).split('.')[0]
        sample['name'] = img_nm
        if(self.gt_txt_dir!=None):
            assert(self.in_box_format!=None)
            ytdir = os.path.join(self.gt_txt_dir,self.gt_txt_name_lambda(img_nm)) if(self.gt_txt_name_lambda)else os.path.join(self.gt_txt_dir,img_nm)
            try:
                boxs, texts = self.read_boxs(ytdir)
            except Exception as ex:
                print("Err when reading GT box at {}.".format(ytdir))
                raise ex
                
            if(not isinstance(boxs,type(None))):
                if(self.normalize): boxs = np_box_nor(boxs,org_shape,self.in_box_format)
                elif(sample['image'].shape[0:2]!=org_shape):
                    boxs = np_box_resize(boxs,org_shape,sample['image'].shape[0:2],self.in_box_format)
                if(self.in_box_format!=self.out_box_format):
                    boxs = np_box_transfrom(boxs,self.in_box_format,self.out_box_format)
                sample['box']= boxs
                sample['box_format']=self.out_box_format

            if(not isinstance(texts,type(None))):sample['text']=texts

        if(self.gt_mask_dir):
            ypdir = os.path.join(self.gt_mask_dir,self.gt_mask_name_lambda(img_nm)) if(self.gt_mask_name_lambda)else os.path.join(self.gt_mask_dir,os.path.basename(self.img_names[idx]))
            ypimg = io.imread(ypdir)
            if(len(ypimg.shape)==2):ypimg = np.expand_dims(ypimg,-1)
            if(not isinstance(self.image_size,type(None)) and ypimg.shape[0:2]!=self.image_size):
                ypimg = transform.resize(ypimg,self.image_size)
            sample['gtmask'] = ypimg

        if(self.transform!=None):
            sample = self.transform(sample)

        return sample

    def get_name(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.img_names[idx]

    def get_by_name(self, fname):
        for i,o in enumerate(self.img_names):
            if(o[-len(fname):].lower()==fname):
                return self[i]
        return None

    def read_boxs(self,fname:str):
        """
        Return boxes,texts
        """
        raise NotImplementedError

