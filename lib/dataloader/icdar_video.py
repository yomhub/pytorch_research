import os
import cv2
import torch
from collections import defaultdict

import xml.etree.ElementTree as ET
def read_boxs(fname:str):
    tree = ET.parse(fname)
    root = tree.getroot()
    pointsxy = defaultdict(list)
    for frame in root:
        fid = int(frame.attrib['ID'])
        for box in frame:
            # box.attrib['Transcription']=='##DONT#CARE##'
            # box.attrib['Quality']=='LOW'
            pointsxy[fid].append(
                [int(box.attrib['ID']),
                int(box[0].attrib['x']),int(box[0].attrib['y']),
                int(box[1].attrib['x']),int(box[1].attrib['y']),
                int(box[2].attrib['x']),int(box[2].attrib['y']),
                int(box[3].attrib['x']),int(box[3].attrib['y'])]
                )
    return pointsxy

class ICDARV():
    def __init__(self, vdo_dir, out_box_format='cxywh', normalized=True):
        self._vdo_dir = vdo_dir
        self._in_box_format = 'xyxy'
        self._gt_name_lambda = lambda x: "%s_GT"%x
        self._vdo_type = ['mp4','avi']
        self._names = [o for o in os.listdir(self._vdo_dir) if o.lower().split('.')[-1] in self._vdo_type]

    def __len__(self):
        return len(self._names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vfile = cv2.VideoCapture(os.path.join(self._vdo_dir,self._names[idx]))
        width  = vfile.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = vfile.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        fps = vfile.get(cv2.CAP_PROP_FPS)
        pointsxy = read_boxs(os.path.join(self._vdo_dir,self._names[idx]+'.xml'))

        sample = {
            'video': vfile,
            'gt_gen': FrameGen(pointsxy,(int(height),int(width))),
            'fps': fps,
            }
        
        return sample

    def get_name(self, index):
        return self._names[index]

class FrameGen(points):
    def __init__(self, points, sizes):
        """
        sizes: (h,w)
        """
        self.points = points
        self.sizes = sizes
    def get_text_part(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        char_gt = np.zeros(self.sizes)
        aff_gt = np.zeros(self.sizes)
        try:
            pg = self.points[idx]
            for o in pg:

        