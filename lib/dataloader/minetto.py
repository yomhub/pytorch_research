import os
import cv2
import torch
import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET

# =====================
from lib.utils.img_hlp import np_box_transfrom

def read_xml(xml_dir,target_fmt:str = 'xywh'):
    target_fmt = target_fmt.lower()
    tree = ET.parse(xml_dir)
    root = tree.getroot()
    box_dict = {}
    txt_dict = {}
    id_dict = {}
    ispoly = 'poly' in target_fmt
    for frame in root:
        fid = int(frame[0].text)
        box_list = []
        txt_list = []
        id_list = []
        for box in frame[2]:
            x,y,w,h=float(box.attrib['x']),float(box.attrib['y']),float(box.attrib['w']),float(box.attrib['h'])
            if(ispoly):
                box_list.append([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
            else:
                box_list.append([x,y,w,h])
            txt_list.append(box.attrib['text'])
            id_list.append(int(box.attrib['id']))
        box_list = np.array(box_list,dtype=np.float32)
        if(len(box_list)>0):
            if(not ispoly):
                box_list = np_box_transfrom(box_list,'xywh',target_fmt)
            box_dict[fid] = box_list
            txt_dict[fid] = txt_list
            id_dict[fid] = id_list
    return box_dict,txt_dict,id_dict

def default_collate_fn(batch):
    return batch[0]

class Minetto():
    """
    Sample dict:
        'video': opencv video object
        'width','height': width, height
        'gt': dictionary, key is frame id start from 0
            item is box array with shape (k,n) where n is (box_id,coordinates)
        'txt': dictionary, key is frame id start from 0
            item is k lenth list
    """
    def __init__(self, vdo_dir, out_box_format='polyxy',include_name=True):
        self._vdo_dir = vdo_dir
        self._names = [o for o in os.listdir(self._vdo_dir) if os.path.exists(os.path.join(self._vdo_dir,o,'PNG'))]
        self._include_name = bool(include_name)
        self._out_box_format = out_box_format.lower()
        self.default_collate_fn = default_collate_fn

    def __len__(self):
        return len(self._names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vfile = cv2.VideoCapture(os.path.join(self._vdo_dir,self._names[idx],'PNG',"""%06d.png"""))
        width  = int(vfile.get(3))
        height = int(vfile.get(4))
        # fps = int(vfile.get(5))
        sample = {
            'video': vfile,
            # 'gt_gen': FrameGen(pointsxy,(int(height),int(width))),
            # 'fps': fps,
            'width':width,
            'height':height,
            'name':self._names[idx],
            }

        if(os.path.exists(os.path.join(self._vdo_dir,self._names[idx],"groundtruth.xml"))):
            box_dict,txt_dict,id_dict = read_xml(os.path.join(self._vdo_dir,self._names[idx],"groundtruth.xml"),self._out_box_format)
            sample['gt']=box_dict
            sample['txt']=txt_dict

        elif(os.path.exists(os.path.join(self._vdo_dir,self._names[idx],"XML"))):
            xml_list = os.listdir(os.path.join(self._vdo_dir,self._names[idx],"XML"))
            box_dict = {}
            txt_dict = {}
            id_dict = {}
            for o in xml_list:
                boxi,txti,idi = read_xml(os.path.join(self._vdo_dir,self._names[idx],"XML",o),self._out_box_format)
                box_dict.update(boxi)
                txt_dict.update(txti)
                id_dict.update(idi)
            sample['gt']=box_dict
            sample['txt']=txt_dict
            sample['id']=id_dict
        
        if(self._include_name):sample['name']=self._names[idx]
        return sample

    def get_name(self, index):
        return self._names[index]