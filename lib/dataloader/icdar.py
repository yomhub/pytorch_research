from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def read_gt(fname):
    f = open(fname, "r")
    lines = f.readlines()
    f.close()
    return [list(map(int, o.split(',')[:-2])) for o in lines],[o.split(',')[-1] for o in lines]


class ICDAR(Dataset):
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

    def __init__(self, img_dir, gt_dir, transform=None):

        imgs = os.listdir(img_dir)
        self.imgs = [os.path.join(img_dirs, name)
                     for name in imgs if
                     (os.path.splitext(name)[-1] == ".jpg" or
                      os.path.splitext(name)[-1] == ".png" or
                      os.path.splitext(name)[-1] == ".bmp")]

        if(gt_dirs):
            gt = os.listdir(gt_dir)
            bsname = imgs[0].split('.')[0]
            prefix = gt[0][:gt[0].find(bsname)]
            suffix = '.'+gt[0].split('.')[-1]
            self.gts = [os.path.join(
                gt_dir, gt_prefix+name.split('.')[0]+suffix) for name in imgs]
        else:
            self.gts = None

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ximg = io.imread(self.imgs[idx])
        gt,txts = read_gt(self.gts[idx]) if(self.gts)else (None,None)

        sample = {'image': ximg}
        if(self.gts):
            sample['gt'] = np.array(gt)
            sample['txt'] = txts

        if(self.transform):
            sample = self.transform(sample)
        return sample
