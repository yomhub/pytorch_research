from __future__ import print_function, division
import os
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from lib.utils.img_hlp import np_box_transfrom,np_box_nor

class BaseDataset(Dataset):
    """
    Args:
        'box_format': string in ['yxyx','xyxy','xywh','cxywh']
            'yxyx': box_cord = [y1,x1,y2,x2]
            'xyxy': box_cord = [x1,y1,x2,y2]
            'xywh': box_cord = [x,y,w,h]
            'cxywh': box_cord = [cx,cy,w,h]
        normalized: True to normalize coordinate
    Outs:
        {
            'image': (h,w,1 or 3) np array.

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

    def __init__(self, img_dir, gt_mask_dir=None, gt_txt_dir=None, in_box_format=None,
        gt_mask_name_lambda=None, gt_txt_name_lambda=None, 
        out_box_format='cxywh', normalized=True, transform=None):
        self.out_box_format = out_box_format.lower() if(out_box_format.lower() in ['yxyx','xyxy','xywh','cxywh'])else 'cxywh'
        if(in_box_format!=None):
            self.in_box_format = in_box_format.lower() if(in_box_format.lower() in ['yxyx','xyxy','xywh','cxywh'])else 'cxywh'
        else:
            self.in_box_format = None
        self.normalize = bool(normalized)
        self.imgdir = img_dir
        self.gt_mask_dir = gt_mask_dir
        self.gt_mask_name_lambda = gt_mask_name_lambda
        self.gt_txt_dir = gt_txt_dir
        self.gt_txt_name_lambda = gt_txt_name_lambda
        img_type = ['jpg','png','bmp']
        self.img_names = [o for o in os.listdir(self.imgdir) if o.lower().split('.')[-1] in img_type]
        self.transform=transform
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'image': io.imread(os.path.join(self.imgdir,self.img_names[idx]))}
        if(self.gt_txt_dir!=None):
            assert(self.in_box_format!=None)
            ytdir = os.path.join(self.gt_txt_dir,self.gt_txt_name_lambda(self.img_names[idx])) if(self.gt_txt_name_lambda)else os.path.join(self.gt_txt_dir,self.img_names[idx])
            boxs, texts = self.read_boxs(ytdir)
            boxs = np_box_transfrom(boxs,self.in_box_format,self.out_box_format)
            if(self.normalize): boxs = np_box_nor(boxs,samplep['image'].shape[-3:-1],self.out_box_format)
            sample['box']=boxs
            sample['box_format']=self.out_box_format
            sample['text']=texts

        if(self.gt_mask_dir):
            ypdir = os.path.join(self.gt_mask_dir,self.gt_mask_name_lambda(self.img_names[idx])) if(self.gt_mask_name_lambda)else os.path.join(self.gt_mask_dir,self.img_names[idx])
            ypimg = io.imread(ypdir)
            if(len(ypimg.shape)==2):ypimg = ypimg.reshape(list(ypimg)+[1])
            sample['gtmask'] = ypimg

        sample = self.transform(sample)

        return sample

    def get_name(self, index):
        return self.img_names[index]

    @abstractmethod
    def read_boxs(self,fname:str):
        pass