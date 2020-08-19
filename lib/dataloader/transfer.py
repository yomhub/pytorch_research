from __future__ import print_function, division
import os
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from . import base

class TransferLearning(base.BaseDataset):
    """
    Transfer learning class
    """
    def __init__(self,img_dir,**params):
        
        super(TransferLearning,self).__init__(img_dir=img_dir,**params)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return None