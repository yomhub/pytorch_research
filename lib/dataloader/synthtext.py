import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transform
# =========================
# from 

__RD_ONLY_MT_MEM = None

def __RD_MAT(mt_dir):
    import scipy.io as scio
    print("load data at {}.\nPlease wait a moment...".format(mt_dir))
    __RD_ONLY_MT_MEM = scio.loadmat(mt_dir)

class SynthText(Dataset):
    def __init__(self, 
    data_dir_path:str, data_file_name:str=None, 
    random_rote_rate=None, istrain:bool=True, 
    image_size=(3, 640, 640), down_rate=2, 
    transform=None):
        # check data path

        data_file_name = "gt.mat" if (data_file_name==None or not isinstance(data_file_name,str))else data_file_name
        self._data_dir_path = data_dir_path

        if(__RD_ONLY_MT_MEM==None):__RD_MAT(os.path.join(self._data_dir_path,data_file_name))


        self._istrain = bool(istrain)
        self._gt = {}
        if istrain:
            self._gt["txt"] = __RD_ONLY_MT_MEM["txt"][0][:-1][:-10000]
            self._gt["imnames"] = __RD_ONLY_MT_MEM["imnames"][0][:-10000]
            self._gt["charBB"] = __RD_ONLY_MT_MEM["charBB"][0][:-10000]
            self._gt["wordBB"] = __RD_ONLY_MT_MEM["wordBB"][0][:-10000]
        else:
            self._gt["txt"] = __RD_ONLY_MT_MEM["txt"][0][-10000:]
            self._gt["imnames"] = __RD_ONLY_MT_MEM["imnames"][0][-10000:]
            self._gt["charBB"] = __RD_ONLY_MT_MEM["charBB"][0][-10000:]
            self._gt["wordBB"] = __RD_ONLY_MT_MEM["wordBB"][0][-10000:]

        self._image_size = image_size
        self._down_rate = down_rate
        self._transform = transform
        self._random_rote_rate = random_rote_rate
    
    def __len__(self):
        return self._gt["txt"].shape[0]
    def _resize(self):
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self._gt["imnames"][idx][0]
        image = Image.open(os.path.join(self.data_dir_path, img_name))
        char_label = self._gt["charBB"][idx].transpose(2, 1, 0)
        if len(self._gt["wordBB"][idx].shape) == 3:
            word_laebl = self._gt["wordBB"][idx].transpose(2, 1, 0)
        else:
            word_laebl = self._gt["wordBB"][idx].transpose(1, 0)[np.newaxis, :]
        txt_label = self._gt["txt"][idx]

        img, char_label, word_laebl = self.resize(image, char_label, word_laebl)