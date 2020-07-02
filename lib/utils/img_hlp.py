from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
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

def np_box_resize(box:np.ndarray,org_size:tuple,new_size:tuple,box_format:str):
    """ Numpy box resize.
        Args: 
        org_size: tuple, (y,x)
        new_size: tuple, (y,x)
    """
    if(box[:,-4:].max()<=1.0): return box # no need for normalized coordinate
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
<<<<<<< HEAD
        scale = np.array((1/tuple[1],1/tuple[0],1/tuple[1],1/tuple[0]))
    else: #'yxyx'
        scale = np.array((1/tuple[0],1/tuple[1],1/tuple[0],1/tuple[1]))
=======
        scale = np.array((1/image_size[1],1/image_size[0],1/image_size[1],1/image_size[0]))
    else: #'yxyx'
        scale = np.array((1/image_size[0],1/image_size[1],1/image_size[0],1/image_size[1]))
>>>>>>> 2e4ce94... Adding dataloder
    ret = box[:,-4:]*scale
    ret = np.clip(ret,0.0,1.0)
    if(box.shape[-1]>4):
        ret = np.concatenate([box[:,0].reshape((-1,1)),ret],axis=-1)
<<<<<<< HEAD
    return box
=======
    return ret
>>>>>>> 2e4ce94... Adding dataloder

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
<<<<<<< HEAD
        min_rate = self.min_size/img_size
        max_rate = self.max_size/img_size if(self.max_size!=None)else np.maximum(min_rate,(1.5,1.5))
        rate = np.random.uniform(min_rate,max_rate,2)
        img_size *= rate
        s_shape = (image.shape[0],int(img_size[0]),int(img_size[1]),image.shape[-1]) if(len(img_size)==4)else (int(img_size[0]),int(img_size[1]),image.shape[-1])
        sample['image'] = image.resize(s_shape)
=======
        min_rate = np.divide(self.min_size,img_size)
        max_rate = np.divide(self.max_size,img_size) if(self.max_size!=None)else np.maximum(min_rate,(1.5,1.5))
        rate = np.random.uniform(min_rate,max_rate,2)
        img_size *= rate
        s_shape = (image.shape[0],int(img_size[0]),int(img_size[1]),image.shape[-1]) if(len(img_size)==4)else (int(img_size[0]),int(img_size[1]),image.shape[-1])
        sample['image'] = np.resize(image,s_shape)
>>>>>>> 2e4ce94... Adding dataloder
        if('box_format' in sample):
            fmt = sample['box_format'].lower()
            box = sample['box']
            if(isinstance(box,list)):
<<<<<<< HEAD
                sample['box'] = [np_box_rescale(o,rate,fmt) if(if(box[:,-4:].max()<=1.0))else o for o in box]
        if('gtmask' in sample):
            sample['gtmask'] = sample['gtmask'].resize(s_shape)
=======
                sample['box'] = [np_box_rescale(o,rate,fmt) if(box[:,-4:].max()<=1.0)else o for o in box]
        if('gtmask' in sample):
            sample['gtmask'] = np.resize(sample['gtmask'],s_shape)
>>>>>>> 2e4ce94... Adding dataloder

        return sample