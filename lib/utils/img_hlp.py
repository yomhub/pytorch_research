from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
The single sample of dataset should include:
    {
        'image': (h,w,1 or 3) or (N,h,w,1 or 3) np array.
        'box': (k,4 or 5) np array or [(k,4 or 5)]*N list of np array.
            where 4 or 5 is (1 or 0)+(box_cord,4)
        'box_format': string in ['yxyx','xyxy','xywh','cxywh']
            'yxyx': box_cord = [y1,x1,y2,x2]
            'xyxy': box_cord = [x1,y1,x2,y2]
            'xywh': box_cord = [x,y,w,h]
            'cxywh': box_cord = [cx,cy,w,h]
        'gtmask': (h,w,1 or 3) or (N,h,w,1 or 3) np array.
    }
"""

# default is __DEF_FORMATS[0]
__DEF_FORMATS = ['cxywh','yxyx','xyxy','xywh']

def np_box_resize(box:np.ndarray,org_size:tuple,new_size,box_format:str):
    """ Numpy box resize.
        Args: 
        org_size: tuple, (y,x)
        new_size: tuple, (y,x) or int for both yx
        box_format: in 'cxywh','yxyx','xyxy','xywh'
    """
    if(box[:,-4:].max()<=1.0): return box # no need for normalized coordinate
    if(isinstance(new_size,int) or isinstance(new_size,float)):new_size = (int(new_size),int(new_size))
    rate = np.divide(new_size,org_size)
    if(box_format.lower() in ['xyxy','xywh','cxywh']):
        rate = np.array((rate[1],rate[0],rate[1],rate[0]))
    else: #'yxyx'
        rate = np.array((rate[0],rate[1],rate[0],rate[1]))
    ret = box[:,-4:]*rate
    if(box.shape[-1]>4):
        ret = np.concatenate([box[:,0].reshape((-1,1)),ret],axis=-1)
    return ret

def np_box_rescale(box:np.ndarray,scale,box_format:str):
    """ Numpy box rescale.
    Args: 
        scale: tuple (y,x) or float
    """
    if(box[:,-4:].max()<=1.0): return box # no need for normalized coordinate
    if(isinstance(scale,float)):scale = (scale,scale)
    scale = np.clip(scale,0.001,10)
    if(box_format.lower() in ['xyxy','xywh','cxywh']):
        scale = np.array((scale[1],scale[0],scale[1],scale[0]))
    else: #'yxyx'
        scale = np.array((scale[0],scale[1],scale[0],scale[1]))
    ret = box[:,-4:]*scale
    if(box.shape[-1]>4):
        ret = np.concatenate([box[:,0].reshape((-1,1)),ret],axis=-1)
    return ret

def np_box_transfrom(box:np.ndarray,src_format:str,dst_format:str)->np.ndarray:
    """
    Box transfrom in ['yxyx','xyxy','xywh','cxywh']
    """
    src_format = src_format.lower() if(src_format.lower() in __DEF_FORMATS)else __DEF_FORMATS[0]
    dst_format = dst_format.lower() if(dst_format.lower() in __DEF_FORMATS)else __DEF_FORMATS[0]
    if(dst_format==src_format):return box
    # convert all to 'cxywh'
    if(src_format=='yxyx'):
        ret = np.stack([
            (box[:,-1]+box[:,-3])/2,#cx
            (box[:,-2]+box[:,-4])/2,#cy
            box[:,-1]-box[:,-3],#w
            box[:,-2]-box[:,-4],#h
            ],axis=-1)
    elif(src_format=='xywh'):
        ret = np.stack([
            box[:,-4]+box[:,-2]*0.5,#cx
            box[:,-3]+box[:,-1]*0.5,#cy
            box[:,-2],#w
            box[:,-1],#h
            ],axis=-1)
    elif(src_format=='xyxy'):
        ret = np.stack([
            (box[:,-2]+box[:,-4])/2,#cx
            (box[:,-1]+box[:,-3])/2,#cy
            box[:,-2]-box[:,-4],#w
            box[:,-1]-box[:,-3],#h
            ],axis=-1)
    # convert from 'cxywh'
    if(dst_format=='yxyx'):
        ret = np.stack([
            ret[:,-3]-ret[:,-1]/2,#y1
            ret[:,-4]-ret[:,-2]/2,#x1
            ret[:,-3]+ret[:,-1]/2,#y2
            ret[:,-4]+ret[:,-2]/2,#x2
            ],axis=-1)
    elif(dst_format=='xywh'):
        ret = np.stack([
            ret[:,-4]-ret[:,-2]/2,#cx
            ret[:,-3]-ret[:,-1]/2,#y
            ret[:,-2],#w
            ret[:,-1],#h
            ],axis=-1)
    elif(dst_format=='xyxy'):
        ret = np.stack([
            ret[:,-4]-ret[:,-2]/2,#x1
            ret[:,-3]-ret[:,-1]/2,#y1
            ret[:,-4]+ret[:,-2]/2,#x2
            ret[:,-3]+ret[:,-1]/2,#y2
            ],axis=-1)

    if(box.shape[-1]>4):
        ret = np.concatenate([box[:,0].reshape((-1,1)),ret],axis=-1)
    return ret

def np_box_nor(box:np.ndarray,image_size:tuple,box_format:str)->np.ndarray:
    """
    Box normalization.
    Args:
        image_size: tuple (y,x)
        box_format: ['yxyx','xyxy','xywh','cxywh']
    """
    if(box_format.lower() in ['xyxy','xywh','cxywh']):
        scale = np.array((1/image_size[1],1/image_size[0],1/image_size[1],1/image_size[0]))
    else: #'yxyx'
        scale = np.array((1/image_size[0],1/image_size[1],1/image_size[0],1/image_size[1]))
    ret = box[:,-4:]*scale
    ret = np.clip(ret,0.0,1.0)
    if(box.shape[-1]>4):
        ret = np.concatenate([box[:,0].reshape((-1,1)),ret],axis=-1)
    return ret

def np_img_resize(img:np.ndarray,new_size=None,base_divisor=None):
    """
    Resize and crop image
    Args: 
        img: ndarray with shape (N,y,x,c) or (y,x,c)
        new_size: tuple, (y,x) or int for both yx, 
            None to disable resize, only use crop function
        base_divisor: if not None, size will crop to N*divisor
            tuple, (y,x) or int for both yx
    Return image
    """
    if(new_size==None): new_size = img.shape[-3:-1]
    if(isinstance(new_size,int) or isinstance(new_size,float)):new_size = (new_size,new_size)
    if(base_divisor!=None):
        new_size = np.ceil(np.divide(new_size,base_divisor))*base_divisor
        new_size = new_size.astype(np.int32)
    s_shape = (img.shape[0],int(new_size[0]),int(new_size[1]),img.shape[-1]) if(len(img)==4)else (int(new_size[0]),int(new_size[1]),img.shape[-1])
    if(new_size[0]!=img.shape[-3] or new_size[1]!=img.shape[-2]): img = np.resize(img,s_shape)
    return img

def cv_score2boxs(score:np.ndarray,threshold=None):
    """
    Convert score 2 box

    """
    if(len(score.shape)>2):
        score = np.clip(score.reshape(score.shape[-3:-1]),0,1)
    else:
        score = np.clip(score.copy(),0,1)

    img_h, img_w = score.shape
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        score.astype(np.uint8), connectivity=4)

    det = []
    mapper = []

    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper

    
class RandomScale(object):
    """Resize randomly the image in a sample.

    Args:
        min_size (tuple or int): Desired output size, tuple for yx, int for both yx.

    """

    def __init__(self, min_size, max_size=None):
        if isinstance(min_size, int):
            self.min_size = (min_size, min_size)
        else:
            assert len(min_size) >= 2
            self.min_size = min_size[0:2]
        if(max_size!=None):
            if isinstance(max_size, int):
                self.max_size = (max_size, max_size)
            else:
                assert len(max_size) >= 2
                self.max_size = max_size[0:2]
        else:
            self.max_size = None

    def __call__(self, sample):
        image = sample['image']
        img_size = image.shape[-3:-1]
        min_rate = np.divide(self.min_size,img_size)
        max_rate = np.divide(self.max_size,img_size) if(self.max_size!=None)else np.maximum(min_rate,(1.5,1.5))
        rate = np.random.uniform(min_rate,max_rate,2)
        img_size *= rate

        s_shape = (image.shape[0],int(img_size[0]),int(img_size[1]),image.shape[-1]) if(len(image.shape)==4)else (int(img_size[0]),int(img_size[1]),image.shape[-1])
        sample['image'] = np.resize(image,s_shape)

        if('box_format' in sample):
            fmt = sample['box_format'].lower()
            box = sample['box']
            if(isinstance(box,list)):
                sample['box'] = [np_box_rescale(o,rate,fmt) if(if(box[:,-4:].max()<=1.0))else o for o in box]
            else:
                sample['box'] = np_box_rescale(box,rate,fmt)
        if('gtmask' in sample):
            sample['gtmask'] = sample['gtmask'].resize(s_shape)

        return sample