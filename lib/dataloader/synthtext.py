import os
import cv2
import numpy as np
# from skimage import io
from skimage import transform as TR
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from lib.utils.img_hlp import *

RD_ONLY_MT_MEM = None
def default_collate_fn(batch):
    ret = {}
    for key,value in batch[0].items():
        if(key.lower() in ['box_format']):
            # single
            ret[key]=value[0] if(isinstance(value,list))else value
        elif(key.lower() in ['box','text','chbox','name'] or type(value) in [str]):
            # List
            ret[key]=[d[key] for d in batch]
        elif(isinstance(value,np.ndarray)):
            ret[key]=np.stack([d[key] for d in batch],0)
        else:
            # stack
            ret[key]=torch.stack([d[key] for d in batch],0)
    return ret

def _rd_mat(mt_dir):
    global RD_ONLY_MT_MEM
    import scipy.io as scio
    RD_ONLY_MT_MEM = scio.loadmat(mt_dir)

class SynthText(Dataset):
    def __init__(self, 
    data_dir_path:str, data_file_name:str=None, 
    mask_dir_path:str=None,
    random_rote_rate=None, istrain:bool=True, 
    image_size=None, transform=None):
        # check data path
        global RD_ONLY_MT_MEM
        data_file_name = "gt.mat" if (data_file_name==None or not isinstance(data_file_name,str))else data_file_name
        self.data_dir_path = data_dir_path

        if(RD_ONLY_MT_MEM==None):_rd_mat(os.path.join(self.data_dir_path,data_file_name))
        self.mask_dir_path = None
        if(mask_dir_path and os.path.exists(mask_dir_path)):
            self.mask_dir_path = mask_dir_path
        elif(os.path.exists(os.path.join(data_dir_path,'mask'))):
            self.mask_dir_path = os.path.join(data_dir_path,'mask')

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
        self.image_size = None
        if(type(image_size) in [list,tuple]): 
            self.image_size = image_size
        elif(type(image_size) in [int,float]):
            self.image_size = (image_size,image_size)

        self.transform = transform
        self.random_rote_rate = random_rote_rate
        self.x_input_function = x_input_function
        self.y_input_function = y_input_function
        self.default_collate_fn = default_collate_fn

    def __len__(self):
        return self.gt["txt"].shape[0]
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.gt["imnames"][idx][0]
        img = io.imread(os.path.join(self.data_dir_path, img_name))
        org_shape = img.shape[:2]

        char_label = self.gt["charBB"][idx].transpose(2, 1, 0).astype(np.float32)
        if(len(self.gt["wordBB"][idx].shape)==3):
            word_label = self.gt["wordBB"][idx].transpose(2, 1, 0).astype(np.float32)
        else:
            word_label = np.expand_dims(self.gt["wordBB"][idx],-1).transpose(2, 1, 0).astype(np.float32)
        txt_label = self.gt["txt"][idx]

        if(self.image_size and self.image_size!=org_shape):
            img = TR.resize(img,self.image_size,preserve_range=True)
            char_label = np_box_resize(char_label,org_shape,self.image_size,'polyxy')
            char_label = np.clip(char_label,0,max(self.image_size))
            word_label = np_box_resize(word_label,org_shape,self.image_size,'polyxy')
            word_label = np.clip(word_label,0,max(self.image_size))

        if self.random_rote_rate:
            angel = np.random.randint(0-self.random_rote_rate, self.random_rote_rate)
            img, M = cv_rotate(img,angel)
            char_label = np_polybox_rotate(char_label,M)
            word_label = np_polybox_rotate(word_label,M)

        if(self.mask_dir_path):
            maskdir = os.path.join(self.mask_dir_path, img_name)
            maskdir,fname = os.path.split(maskdir)
            char_gt = io.imread(os.path.join(maskdir, 'ch_{}'.format(fname)))
            aff_gt = io.imread(os.path.join(maskdir, 'af_{}'.format(fname)))
            char_gt = char_gt[:,:,0].astype(np.float32)/255.0
            aff_gt = aff_gt[:,:,0].astype(np.float32)/255.0
            
            line_boxes = []
            for txt in txt_label:
                for strings in txt.split("\n"):
                    for string in strings.split(" "):
                        char_boxes = [char_label[char_index+i] for i in range(len(string))]
                        char_index += len(string)
                        line_boxes.append(np.array(char_boxes))
        else:
            char_gt = np.zeros(img.shape[:2],dtype=np.float32)
            aff_gt = np.zeros(img.shape[:2],dtype=np.float32)

            line_boxes = []
            char_index = 0
            word_index = 0
            high, width = img.shape[:2]
                
            char_label_xyxy = np_polybox_minrect(char_label,'polyxy')
            dxy = char_label_xyxy[:,2]-char_label_xyxy[:,0]
            for txt in txt_label:
                for strings in txt.split("\n"):
                    for string in strings.split(" "):
                        if string == "":
                            continue
                        char_boxes = []
                        for char in string:
                            char_boxes.append(char_label[char_index])
                            deta_x, deta_y = dxy[char_index].astype(np.int16)
                            box = char_label_xyxy[char_index]
                            min_x = max(int(box[0,0]),0)
                            min_y = max(int(box[0,1]),0)
                            if(deta_x <= 0 or deta_y <= 0):
                                char_index += 1
                                continue
                            if(min_x+deta_x>width):
                                deta_x = width-min_x
                            if(min_y+deta_y>high):
                                deta_y = high-min_y

                            try:
                                gaussian = cv_gaussian_kernel_2d(kernel_size=(deta_y, deta_x))
                                # gaussian = np_2d_gaussian((deta_y, deta_x))
                                res = aff_gaussian(gaussian, box, char_label[char_index], deta_y, deta_x)
                                max_v = np.max(res)
                            except:
                                char_index += 1
                                continue
                                                    
                            if(max_v > 0):
                                res /= max_v
                                sub_mask = char_gt[min_y:min_y+res.shape[0],min_x:min_x+res.shape[1]]
                                if(sub_mask.shape!=res.shape):
                                    print("{}_{}".format(sub_mask.shape,res.shape))
                                sub_mask = np.where(sub_mask>res,sub_mask,res)
                                char_gt[min_y:min_y+res.shape[0],min_x:min_x+res.shape[1]] = sub_mask                           
                            char_index += 1
                        word_index += 1
                        line_boxes.append(np.array(char_boxes))
            for char_boxes in line_boxes:
                if(char_boxes.shape[0]<=1):
                    continue
                affine_boxes = create_affine_boxes(char_boxes.reshape(-1,4,2))
                for points in affine_boxes:
                    box = np_polybox_minrect(points,'polyxy')
                    deta_x,deta_y = (box[2]-box[0]).astype(np.int16)
                    min_x = max(int(box[0,0]),0)
                    min_y = max(int(box[0,1]),0)
                    if(deta_x <= 0 or deta_y <= 0):
                        continue
                    if(min_x+deta_x>width):
                        deta_x = width-min_x
                    if(min_y+deta_y>high):
                        deta_y = high-min_y
                    try:
                        gaussian = cv_gaussian_kernel_2d(kernel_size=(deta_y, deta_x))
                        # gaussian = np_2d_gaussian((deta_y, deta_x))
                        res = aff_gaussian(gaussian, box, points,  deta_y, deta_x)
                        max_v = np.max(res)
                    except:
                        continue

                    if(max_v > 0):
                        res /= max_v
                        sub_mask = aff_gt[min_y:min_y+res.shape[0],min_x:min_x+res.shape[1]]
                        if(sub_mask.shape!=res.shape):
                            print("{}_{}".format(sub_mask.shape,res.shape))
                        sub_mask = np.where(sub_mask>res,sub_mask,res)
                        aff_gt[min_y:min_y+res.shape[0],min_x:min_x+res.shape[1]] = sub_mask

        # reshape to (h,w,1)
        char_gt = np.expand_dims(char_gt,-1)
        aff_gt = np.expand_dims(aff_gt,-1)

        sample = {
            'image': img,
            'char_gt': char_gt,
            'aff_gt': aff_gt,
            'box': word_label,
            'chbox': line_boxes,
            'box_format': 'polyxy',
            'name':img_name,
            # 'affine_boxes': affine_boxes,
            # 'line_boxes': line_boxes,
            # 'char_label': char_label,
        }

        return sample


def x_input_function(sample,th_device): 
    x = sample['image'] if(isinstance(sample,dict))else sample
    return to_torch(x,th_device).permute(0,3,1,2)

def y_input_function(sample,th_device): 
    char_gt = sample['char_gt'] if(isinstance(sample,dict))else sample[0]
    aff_gt = sample['aff_gt'] if(isinstance(sample,dict))else sample[1]

    return to_torch(char_gt,th_device).permute(0,3,1,2), to_torch(aff_gt,th_device).permute(0,3,1,2)

def create_affine_boxes(boxs):
    """
    boxs: (N,4,2) with (x,y)
    return
    """
    sp = boxs[:-1]
    top_center = (sp[:,1]+sp[:,0])/2.0
    bot_center = (sp[:,3]+sp[:,2])/2.0
    up_cent_sp = bot_center + (top_center-bot_center)*0.75
    dw_cent_sp = bot_center + (top_center-bot_center)*0.25
    ep = boxs[1:]
    top_center = (ep[:,1]+ep[:,0])/2.0
    bot_center = (ep[:,3]+ep[:,2])/2.0
    up_cent_ep = bot_center + (top_center-bot_center)*0.75
    dw_cent_ep = bot_center + (top_center-bot_center)*0.25

    return np.stack([up_cent_sp,up_cent_ep,dw_cent_ep,dw_cent_sp],axis=1)

def aff_gaussian(gaussian, box, pts, deta_x, deta_y):
    box = box.astype(np.float32)
    pts = pts.astype(np.float32)
    de_x, de_y = box[0]
    box = box - [de_x, de_y]
    pts = pts - [de_x, de_y]
    M = cv2.getPerspectiveTransform(box, pts)
    res = cv2.warpPerspective(gaussian, M, (deta_y, deta_x))
    return res