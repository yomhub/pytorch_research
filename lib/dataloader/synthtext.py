import os
import cv2
import numpy as np
# from skimage import io
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# =========================
# from 

RD_ONLY_MT_MEM = None

def _rd_mat(mt_dir):
    global RD_ONLY_MT_MEM
    import scipy.io as scio
    RD_ONLY_MT_MEM = scio.loadmat(mt_dir)

class SynthText(Dataset):
    def __init__(self, 
    data_dir_path:str, data_file_name:str=None, 
    random_rote_rate=None, istrain:bool=True, 
    image_size=(640, 640), down_rate=2, 
    transform=None):
        # check data path
        global RD_ONLY_MT_MEM
        data_file_name = "gt.mat" if (data_file_name==None or not isinstance(data_file_name,str))else data_file_name
        self._data_dir_path = data_dir_path

        if(RD_ONLY_MT_MEM==None):_rd_mat(os.path.join(self._data_dir_path,data_file_name))


        self._istrain = bool(istrain)
        self._gt = {}
        if istrain:
            self._gt["txt"] = RD_ONLY_MT_MEM["txt"][0][:-1][:-10000]
            self._gt["imnames"] = RD_ONLY_MT_MEM["imnames"][0][:-10000]
            self._gt["charBB"] = RD_ONLY_MT_MEM["charBB"][0][:-10000]
            self._gt["wordBB"] = RD_ONLY_MT_MEM["wordBB"][0][:-10000]
        else:
            self._gt["txt"] = RD_ONLY_MT_MEM["txt"][0][-10000:]
            self._gt["imnames"] = RD_ONLY_MT_MEM["imnames"][0][-10000:]
            self._gt["charBB"] = RD_ONLY_MT_MEM["charBB"][0][-10000:]
            self._gt["wordBB"] = RD_ONLY_MT_MEM["wordBB"][0][-10000:]

        # (x,y)
        self._image_size = image_size if(isinstance(image_size,tuple) or isinstance(image_size,list))else (image_size,image_size)
        self._down_rate = down_rate
        self._transform = transform
        self._random_rote_rate = random_rote_rate
    
    def __len__(self):
        return self._gt["txt"].shape[0]

    def _resize(self,image,char_label,word_laebl):
        # (x,y)
        org_size = image.size
        np.divide()
        image = image.resize(self._image_size)
        return image,char_label,word_laebl

        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}
        img_name = self._gt["imnames"][idx][0]
        image = Image.open(os.path.join(self._data_dir_path, img_name))
        # (N,4,2) for [char idx][4 points idx](x,y)
        char_label = self._gt["charBB"][idx].transpose(2, 1, 0)
        if len(self._gt["wordBB"][idx].shape) == 3:
            word_laebl = self._gt["wordBB"][idx].transpose(2, 1, 0)
        else:
            word_laebl = self._gt["wordBB"][idx].transpose(1, 0)[np.newaxis, :]
        txt_label = self._gt["txt"][idx]

        img, char_label, word_laebl = self._resize(image, char_label, word_laebl)
        txt_str = ''
        for o in txt_label:
            txt_str += o.replace(' ', '').replace('\n', '')
        
        char_label



        sample['image'] = image

        return sample

if __name__ == "__main__":
    sydata = SynthText(data_dir_path=r"D:\development\SynthText")
    sydata[0]
    loader = DataLoader(sydata,10)
    i=0
    for o in loader:
        print(o['image'].shape)
    pass