from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Total(Dataset):
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
    """

    def __init__(self, root_dir, gt_format='mask', transform=None):
        self.gt_format = 'mask' if(gt_format.lower() == 'mask')else 'gtbox'
        self.imgdir = os.path.join(root_dir, 'Images')
        self.ypdir = os.path.join(root_dir, 'gt_pixel')
        self.ytdir = os.path.join(root_dir, 'gt_txt')

        self.train_img_names = []
        for root, dirs, files in os.walk(os.path.join(self.imgdir, 'Train')):
            self.train_img_names += [os.path.join('Train', name) for name in files if (os.path.splitext(name)[-1] == ".jpg" or
                                                                                       os.path.splitext(name)[-1] == ".png" or
                                                                                       os.path.splitext(name)[-1] == ".bmp")]
        self.test_img_names = []
        for root, dirs, files in os.walk(os.path.join(self.imgdir, 'Test')):
            self.test_img_names += [os.path.join('Test', name) for name in files if (os.path.splitext(name)[-1] == ".jpg" or
                                                                                     os.path.splitext(name)[-1] == ".png" or
                                                                                     os.path.splitext(name)[-1] == ".bmp")]

        self.all_img_dir = [name for name in self.train_img_names] +\
            [name for name in self.test_img_names]
        
        self.transform=transform
        
    def __len__(self):
        return len(self.all_img_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ytdirs = os.path.join(self.ytdir,self.all_img_dir[idx])
        ximgs = io.imread(os.path.join(self.imgdir,self.all_img_dir[idx])) 
        ypimgs = io.imread(os.path.join(self.ypdir,self.all_img_dir[idx]))

        sample = {'image': ximgs, 'gtmask': ypimgs, }
        sample = self.transform(sample)

        return sample
        