import os
import cv2
import numpy as np
# from skimage import io
from skimage import transform as TR
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from lib.utils.img_hlp import np_corp_points, to_torch
# =========================
try:
    import lib.dataloader.datautils as datautils
    # import lib.dataloader.transutils
except:
    from . import datautils
    # from . import transutils

RD_ONLY_MT_MEM = None
def default_collate_fn(batch):
    ret = {}
    for key,value in batch[0].items():
        if(key.lower() in ['box','text']):
            ret[key]=[d[key] for d in batch]
        elif(key.lower() in ['box_format']):
            ret[key]=value
        else:
            ret[key]=torch.stack([torch.from_numpy(d[key])if(isinstance(d[key],np.ndarray))else d[key] for d in batch],0)
    return ret

def _rd_mat(mt_dir):
    global RD_ONLY_MT_MEM
    import scipy.io as scio
    RD_ONLY_MT_MEM = scio.loadmat(mt_dir)

class SynthText(Dataset):
    def __init__(self, 
    data_dir_path:str, data_file_name:str=None, 
    random_rote_rate=None, istrain:bool=True, 
    image_size=(3,640, 640), down_rate=2, 
    transform=None):
        # check data path
        global RD_ONLY_MT_MEM
        data_file_name = "gt.mat" if (data_file_name==None or not isinstance(data_file_name,str))else data_file_name
        self.data_dir_path = data_dir_path

        if(RD_ONLY_MT_MEM==None):_rd_mat(os.path.join(self.data_dir_path,data_file_name))


        self.istrain = bool(istrain)
        self.gt = {}
        if istrain:
            self.gt["txt"] = RD_ONLY_MT_MEM["txt"][0][:-1][:-10000]
            self.gt["imnames"] = RD_ONLY_MT_MEM["imnames"][0][:-10000]
            self.gt["charBB"] = RD_ONLY_MT_MEM["charBB"][0][:-10000]
            self.gt["wordBB"] = RD_ONLY_MT_MEM["wordBB"][0][:-10000]
        else:
            self.gt["txt"] = RD_ONLY_MT_MEM["txt"][0][-10000:]
            self.gt["imnames"] = RD_ONLY_MT_MEM["imnames"][0][-10000:]
            self.gt["charBB"] = RD_ONLY_MT_MEM["charBB"][0][-10000:]
            self.gt["wordBB"] = RD_ONLY_MT_MEM["wordBB"][0][-10000:]

        # (x,y)
        self.image_size = image_size if(isinstance(image_size,tuple) or isinstance(image_size,list))else (3,image_size,image_size)
        self.down_rate = down_rate
        self.transform = transform
        self.random_rote_rate = random_rote_rate
        self.x_input_function = x_input_function
        self.y_input_function = y_input_function
        self.default_collate_fn = default_collate_fn

    def __len__(self):
        return self.gt["txt"].shape[0]

    def resize(self, image, char_label, word_laebl):
        w, h = image.size
        img = np.zeros(self.image_size)
        rate = self.image_size[2] / self.image_size[1]
        rate_pic = w / h
        
        if rate_pic > rate:
            resize_h = int(self.image_size[2] / rate_pic)
            image = image.resize((self.image_size[2], resize_h), Image.ANTIALIAS)
            image = np.array(image)
            if self.image_size[0] == 3:
                if len(image.shape) == 2:
                    image = np.tile(image, (3, 1, 1))
                else:
                    image = image.transpose((2, 0, 1))

            img[:,:resize_h,:] = image
            char_label = char_label * (resize_h / h)
            word_laebl = word_laebl * (resize_h / h)
        else:
            resize_w = int(rate_pic * self.image_size[1])
            image = image.resize((resize_w, self.image_size[1]), Image.ANTIALIAS)
            image = np.array(image)
            if self.image_size[0] == 3:
                if len(image.shape) == 2:
                    image = np.tile(image, (3, 1, 1))
                else:
                    image = image.transpose((2, 0, 1))

            img[:,:,:resize_w] = np.array(image)
            char_label = char_label * (resize_w / w)
            word_laebl = word_laebl * (resize_w / w)
        return img, char_label, word_laebl

        
    def __getitem__(self, idx):
        img_name = self.gt["imnames"][idx][0]
        image  = Image.open(os.path.join(self.data_dir_path, img_name))
        char_label = self.gt["charBB"][idx].transpose(2, 1, 0)
        if len(self.gt["wordBB"][idx].shape) == 3:
            word_laebl = self.gt["wordBB"][idx].transpose(2, 1, 0)
        else:
            word_laebl = self.gt["wordBB"][idx].transpose(1, 0)[np.newaxis, :]
        txt_label = self.gt["txt"][idx]

        img, char_label, word_laebl = self.resize(image, char_label, word_laebl)

        if self.random_rote_rate:
            angel = random.randint(0-self.random_rote_rate, self.random_rote_rate)
            img, M = datautils.rotate(angel, img)

        char_gt = np.zeros((int(self.image_size[1]), int(self.image_size[2])))
        aff_gt = np.zeros((int(self.image_size[1]), int(self.image_size[2])))

        
        line_boxes = []
        char_index = 0
        word_index = 0
        for txt in txt_label:
            for strings in txt.split("\n"):
                for string in strings.split(" "):
                    if string == "":
                        continue
                    char_boxes = []
                    for char in string:
                        x0, y0 = char_label[char_index][0]
                        x1, y1 = char_label[char_index][1]
                        x2, y2 = char_label[char_index][2]
                        x3, y3 = char_label[char_index][3]
                        
                        if self.random_rote_rate:
                            x0, y0 = datautils.rotate_point(M, x0, y0)
                            x1, y1 = datautils.rotate_point(M, x1, y1)
                            x2, y2 = datautils.rotate_point(M, x2, y2)
                            x3, y3 = datautils.rotate_point(M, x3, y3)
                        
                        x0, y0, x1, y1, x2, y2, x3, y3 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), int(round(x3)), int(round(y3))
                        char_boxes.append([x0, y0, x1, y1, x2, y2, x3, y3])
                        box, deta_x, deta_y = datautils.find_min_rectangle([x0, y0, x1, y1, x2, y2, x3, y3])
                        if deta_x <= 0 or deta_x >= self.image_size[2] or deta_y <= 0 or deta_y >= self.image_size[1]:
                            # print(idx, deta_x, deta_y)
                            char_index += 1
                            continue
                        try:
                            gaussian = datautils.gaussian_kernel_2d_opencv(kernel_size=(deta_y, deta_x))
                            pts = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                            res = datautils.aff_gaussian(gaussian, box, pts, deta_y, deta_x)
                        except:
                            char_index += 1
                            continue
                        
                        min_x = min(x0, x1, x2, x3)
                        min_y = min(y0, y1, y2, y3)

                        if np.max(res) > 0:
                            mx = 1 / np.max(res)
                            res = mx * res
                            gh, gw = res.shape
                            for th in range(gh):
                                for tw in range(gw):
                                    if 0 < min_y+th < char_gt.shape[0] and 0 < min_x+tw < char_gt.shape[1]:
                                        try:
                                            char_gt[min_y+th, min_x+tw] = max(char_gt[min_y+th, min_x+tw], res[th, tw])
                                        except:
                                            print(idx, min_y+th, min_x+tw)
                            
                        char_index += 1
                    word_index += 1
                    line_boxes.append(char_boxes)
        affine_boxes = []
        for char_boxes in line_boxes:
            affine_boxes.extend(datautils.create_affine_boxes(char_boxes))
            for points in affine_boxes:
                x0, y0, x1, y1, x2, y2, x3, y3 = points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7]
                box, deta_x, deta_y = datautils.find_min_rectangle(points)
                if deta_x <= 0 or deta_x >= self.image_size[2] or deta_y <= 0 or deta_y >= self.image_size[1]:
                    continue
                try:
                    gaussian = datautils.gaussian_kernel_2d_opencv(kernel_size=(deta_y, deta_x))
                    pts = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                    res = datautils.aff_gaussian(gaussian, box, pts,  deta_y, deta_x)
                except:
                    continue
                min_x = min(x0, x1, x2, x3)
                min_y = min(y0, y1, y2, y3)

                if np.max(res) > 0:
                    mx = 1 / np.max(res)
                    res = mx * res
                    gh, gw = res.shape
                    for th in range(gh):
                        for tw in range(gw):
                            if 0 < min_y+th < aff_gt.shape[0] and 0 < min_x+tw < aff_gt.shape[1]:
                                try:
                                    aff_gt[min_y+th, min_x+tw] = max(aff_gt[min_y+th, min_x+tw], res[th, tw])
                                except:
                                    print(idx, min_y+th, min_x+tw)
        sample = {
            # 'image': img if(self.transform)else self.transform(img),
            'image': np.transpose(img,(1,2,0)),
            'char_gt': np.expand_dims(TR.resize(char_gt, (int(self.image_size[1]/self.down_rate), int(self.image_size[2]/self.down_rate))),axis=0),
            'aff_gt': np.expand_dims(TR.resize(aff_gt, (int(self.image_size[1]/self.down_rate), int(self.image_size[2]/self.down_rate))),axis=0),
            # 'affine_boxes': affine_boxes,
            # 'line_boxes': line_boxes,
            # 'char_label': char_label
        }

        return sample


def x_input_function(sample,th_device): 
    x = sample['image'] if(isinstance(sample,dict))else sample
    return to_torch(x,th_device).permute(0,3,1,2)

def y_input_function(sample,th_device): 
    char_gt = sample['char_gt'] if(isinstance(sample,dict))else sample[0]
    aff_gt = sample['aff_gt'] if(isinstance(sample,dict))else sample[1]

    return to_torch(char_gt,th_device), to_torch(aff_gt,th_device)
