import os
import cv2
import torch
import numpy as np
from collections import defaultdict
# =====================
from lib.utils.img_hlp import np_box_transfrom

import xml.etree.ElementTree as ET
def read_xml(fname:str,target_fmt:str='xyxy',exclude_boxid:bool=False):
    target_fmt = target_fmt.lower()
    tree = ET.parse(fname)
    root = tree.getroot()
    box_dict = {}
    txt_dict = {}
    id_dict = {}
    ispoly = 'poly' in target_fmt
    for frame in root:
        fid = int(frame.attrib['ID'])
        box_list = []
        txt_list = []
        id_list = []
        for box in frame:
            # box.attrib['Quality'] in ['LOW','MODERATE','HIGH']
            if(box.attrib['Transcription'] == '##DONT#CARE##' or box.attrib['Quality'] in ['LOW','MODERATE']):
                txt_list.append('#')
            else:
                txt_list.append(box.attrib['Transcription'])
            id_list.append(int(box.attrib['ID']))
            
            box_list.append(np.array((int(box[0].attrib['x']),int(box[0].attrib['y']),
                int(box[1].attrib['x']),int(box[1].attrib['y']),
                int(box[2].attrib['x']),int(box[2].attrib['y']),
                int(box[3].attrib['x']),int(box[3].attrib['y']),
                )).reshape(-1,2))

        box_list = np.array(box_list,dtype=np.float32)
        if(len(box_list)>0):
            if(not ispoly):
                box_list = np_box_transfrom(box_list,'polyxy',target_fmt)
            box_dict[fid] = box_list
            txt_dict[fid] = txt_list
            id_dict[fid] = id_list

    return box_dict,txt_dict,id_dict

def default_collate_fn(batch):
    return batch[0]

class ICDARV():
    def __init__(self, vdo_dir, out_box_format='polyxy', normalized=True, include_name=False, exclude_boxid:bool=False):
        self._vdo_dir = vdo_dir
        self._gt_name_lambda = lambda x: "%s_GT"%x
        self._vdo_type = ['mp4','avi']
        self._names = [o for o in os.listdir(self._vdo_dir) if o.lower().split('.')[-1] in self._vdo_type]
        self._include_name = bool(include_name)
        self._out_box_format = out_box_format.lower()
        self.exclude_boxid = exclude_boxid
        self.default_collate_fn = default_collate_fn

    def __len__(self):
        return len(self._names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vfile = cv2.VideoCapture(os.path.join(self._vdo_dir,self._names[idx]))
        width  = int(vfile.get(3))
        height = int(vfile.get(4))
        fps = int(vfile.get(5))
        sample = {
            'video': vfile,
            'fps': fps,
            'width':width,
            'height':height,
            'name':self._names[idx].split('.')[0],
            }

        if(os.path.exists(os.path.join(self._vdo_dir,self._names[idx].split('.')[0]+'_GT.xml'))):
            box_dict,txt_dict,id_dict = read_xml(
                os.path.join(self._vdo_dir,self._names[idx].split('.')[0]+'_GT.xml'),
                self._out_box_format)
            sample['gt']=box_dict
            sample['txt']=txt_dict
            sample['id']=id_dict
        
        if(self._include_name):sample['name']=self._names[idx]
        return sample

    def get_name(self, index):
        return self._names[index]

    def get_by_name(self, fname):
        fname=fname.lower()
        for i,o in enumerate(self._names):
            if(o[-len(fname):].lower()==fname):
                return self[i]
        return None

        