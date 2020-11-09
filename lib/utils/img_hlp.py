from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
import math
from collections import Iterable
import Polygon as plg
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, Point
# ======================
from . import log_hlp

def to_torch(img,th_device):
    if(isinstance(img,np.ndarray)):
        img = torch.from_numpy(img)
    if(len(img.shape)==2):
        img = torch.reshape(img,(1,1,img.shape[0],img.shape[1]))
    elif(len(img.shape)==3):
        img = torch.reshape(img,tuple([1]+list(img.shape)))

    return img.type(torch.FloatTensor).to(th_device)

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
    CV image: ndarray with (h,w,c)
    CV format 4 points: (x,y) in 
    +------------> x
    | 
    | p0------p1
    | |        |
    | |        |
    | p3------p2
    y
"""

# default is __DEF_FORMATS[0]
__DEF_FORMATS = ['cxywh','yxyx','xyxy','xywh','polyxy','polyyx']

def np_box_resize(box:np.ndarray,org_size:tuple,new_size,box_format:str):
    """ Numpy box resize.
        Args: 
        org_size: tuple, (y,x)
        new_size: tuple, (y,x) or int for both yx
        box_format: in 'cxywh','yxyx','xyxy','xywh','polyxy','polyyx'
    """
    if(not isinstance(new_size,Iterable)):
        new_size = (int(new_size),int(new_size))
    if(org_size==new_size):return box
    return np_box_rescale(box,np.divide(new_size,org_size),box_format)

def np_box_rescale(box,scale,box_format:str):
    """ Numpy box rescale.
    Args: 
        box: (N,4 or 5) for rectangle
             (k,2),(N,k*2),(N,k,2) for polygon
        scale: tuple (y,x) or float
        box_format: __DEF_FORMATS for rectangle
            'polyxy','polyyx' for polygon
    """
    if(not isinstance(box,np.ndarray)):box = np.array(box)
    if(box.dtype==np.object):
        tmp = [np_box_rescale(o,scale,box_format) for o in box]
        return np.array(tmp)
    if(box[:,-(box.shape[-1]//2*2):].max()<=1.0): 
        return box # no need for normalized coordinate
    if(not isinstance(scale,Iterable)):scale = (scale,scale)
    scale = np.clip(scale,0.001,10)
    if(box_format.lower() in ['xyxy','xywh','cxywh','polyxy']):
        scale = np.array([scale[1],scale[0]]*(box.shape[-1]//2))
    else: #'yxyx'
        scale = np.array([scale[0],scale[1]]*(box.shape[-1]//2))

    ret = box[:,-(box.shape[-1]//2*2):]*scale if(len(box.shape)==2)else box*scale
    if(len(box.shape)==2 and box.shape[-1]%2):
        ret = np.concatenate([box[:,0].reshape((-1,1)),ret],axis=-1)
    return ret

def np_polybox_minrect(cv_polybox,box_format:str='polyxy'):
    """
    Find minimum rectangle for cv ploy box
        cv_polybox: ((Box number), point number, 2)
        box_format: 'polyxy' or 'polyyx'
    Return: ((Box number),4,2) in 'polyxy' or 'polyyx'
    """
    single_box = False
    if(len(cv_polybox.shape)==2):
        cv_polybox = np.expand_dims(cv_polybox,0)
        single_box = True
    if(cv_polybox.dtype==np.object):
        dim_0s_min = []
        dim_0s_max = []
        dim_1s_min = []
        dim_1s_max = []
        for bxs in cv_polybox:
            dim_0s_min.append(bxs[:,0].min())
            dim_0s_max.append(bxs[:,0].max())
            dim_1s_min.append(bxs[:,1].min())
            dim_1s_max.append(bxs[:,1].max())
        dim_0s_min = np.array(dim_0s_min).reshape(-1,1)
        dim_0s_max = np.array(dim_0s_max).reshape(-1,1)
        dim_1s_min = np.array(dim_1s_min).reshape(-1,1)
        dim_1s_max = np.array(dim_1s_max).reshape(-1,1)
    else:
        dim_0s = cv_polybox[:,:,0]
        dim_1s = cv_polybox[:,:,1]
        dim_0s_min = dim_0s.min(axis=-1).reshape(-1,1)
        dim_0s_max = dim_0s.max(axis=-1).reshape(-1,1)
        dim_1s_min = dim_1s.min(axis=-1).reshape(-1,1)
        dim_1s_max = dim_1s.max(axis=-1).reshape(-1,1)
    
    if(box_format.lower()=='polyyx'):
        ret = np.concatenate((
            dim_0s_min,dim_1s_min,
            dim_0s_min,dim_1s_max,
            dim_0s_max,dim_1s_max,
            dim_0s_max,dim_1s_min,
            ),axis=-1).reshape(-1,4,2)
    else:    
        ret = np.concatenate((
            dim_0s_min,dim_1s_min,
            dim_0s_max,dim_1s_min,
            dim_0s_max,dim_1s_max,
            dim_0s_min,dim_1s_max,
        ),axis=-1).reshape(-1,4,2)
    return ret[0] if(single_box)else ret

def np_polybox_rotate(cv_polybox,M):
    """
    Rotate polybox
        cv_polybox: ((Box number), point number, 2) in (x,y)
        M: (2*3) rotate matrix
    Return: 
        cv_polybox: ((Box number), point number, 2)
    """
    M = M[:2,:3]
    len2 = False
    if(len(cv_polybox.shape)==2):
        cv_polybox = np.expand_dims(cv_polybox,0)
        len2 = True
    elif(cv_polybox.dtype==np.object):
        tmp = [np_polybox_rotate(o,M) for o in cv_polybox]
        return np.array(tmp)
            
    # (x,y)->(x,y,1)
    cv_polybox = np.pad(cv_polybox,[(0,0),(0,0),(0,1)],constant_values=1)
    # change axis from (boxs,points,3) to (boxs,3,points)
    cv_polybox = np.moveaxis(cv_polybox,-1,1)
    # (2,3) dot (boxs,3,points) = (2,boxs,points)
    ret = M.dot(cv_polybox)
    # change (2,boxs,points) to (boxs,points,2)
    ret = np.moveaxis(ret,0,-1)
    return ret if(not len2)else ret[0]

def np_box_transfrom(box:np.ndarray,src_format:str,dst_format:str)->np.ndarray:
    """
    Box transfrom in ['yxyx','xyxy','xywh','cxywh']
    """
    src_format = src_format.lower()
    dst_format = dst_format.lower()
    if(src_format==dst_format):return box
    assert(src_format in __DEF_FORMATS and dst_format in __DEF_FORMATS)

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
    else:
        ret = box[:,-4:]
         
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
        box_format: ['yxyx','xyxy','xywh','cxywh','polyxy','polyyx']
    """
    if(box_format.lower() in ['xyxy','xywh','cxywh','polyxy']):
        scale = np.array([1/image_size[1],1/image_size[0]]*(box.shape[-1]//2))
    else: #'yxyx'
        scale = np.array([1/image_size[0],1/image_size[1]]*(box.shape[-1]//2))
    ret = box[:,-(box.shape[-1]//2*2):]*scale
    ret = np.clip(ret,0.0,1.0)
    if(box.shape[-1]%2):
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

def np_img_normalize(img:np.ndarray,mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    """
    Normalize image
    Args: 
        img: ndarray with shape (N,y,x,c) or (y,x,c)
        mean: float or tuple in each channel in [0,1]
        variance: float or tuple in each channel in [0,1]
    Return image
    """
    chs = img.shape[-1]
    if(not isinstance(mean,Iterable)):mean = [mean]*chs
    if(not isinstance(variance,Iterable)):variance = [variance]*chs
    img = img.astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)

    return img

def torch_img_normalize(img,mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    """
    Normalize image in torch
    Args: 
        img: tensor with shape (N,y,x,c) or (y,x,c)
        mean: float or tuple in each channel in [0,1]
        variance: float or tuple in each channel in [0,1]
    Return image
    """
    chs = img.shape[-1]
    if(not isinstance(mean,Iterable)):mean = [mean]*chs
    if(not isinstance(variance,Iterable)):variance = [variance]*chs
    img = img.float()
    img -= torch.mul(torch.tensor(mean,dtype=img.dtype),255.0).to(img.device)
    img /= torch.mul(torch.tensor(variance,dtype=img.dtype),255.0).to(img.device)
    return img
    
def np_2d_gaussian(img_size,x_range=(-1.5,1.5),y_range=(-1.5,1.5),sigma:float=0.9,mu:float=0.0):
    """
    Generate gaussian distribution.
    Args: 
        x_range/y_range: tuple, (a,b) or float for (-a,a), 
        sigma/mu: float
        img_size: tuple, (y,x) or int for both yx, 
        Default range 1.5, sigma 0.9, mu 0.0, the value will be 
            above 0.5 at centered 70% 
            above 0.3 at centered 90% 
    Return 2d gaussian distribution in numpy.
    """
    if(not isinstance(img_size,Iterable)):
        img_size = (img_size,img_size)
    if(not isinstance(x_range,Iterable)):
        x_range = (-x_range,x_range)
    if(not isinstance(y_range,Iterable)):
        y_range = (-y_range,y_range)

    dx, dy = np.meshgrid(
        np.linspace(x_range[0],x_range[1],img_size[1],dtype=np.float32), 
        np.linspace(y_range[0],y_range[1],img_size[0],dtype=np.float32))

    d = np.sqrt(dx**2+dy**2)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

def cv_gaussian_kernel_2d(kernel_size = (3, 3)):
    """
    Generate gaussian kernel
    """
    ky = cv2.getGaussianKernel(int(kernel_size[0]), int(kernel_size[0] / 4))
    kx = cv2.getGaussianKernel(int(kernel_size[1]), int(kernel_size[1] / 4))
    return np.multiply(ky, np.transpose(kx))

from scipy import ndimage
# For masked polygon distance 
def cv_gen_gaussian_by_poly(cv_box,img_size=None,centralize:bool=False,v_range:float=1.5,sigma:float=0.9):
    """
    Generate gaussian map by polygon
    Args:
        cv_box: (k,2) polygon box 
        img_size: (y,x) output mask shape, if None, use minium box rectangle
        centralize: whether to centralize the polygon
        v_range: MAX distance of gaussian 
        sigma: sigma
    Return:
        gaussian mask for polygon
    """
    box_rect = np_polybox_minrect(cv_box)
    if(not img_size):
        x0,y0 = box_rect[0]
        x1,y1 = box_rect[2]
        x0 = max(0,int(x0))
        y0 = max(0,int(y0))
        img_size = (int(y1-y0),int(x1-x0))
    elif(not isinstance(img_size,Iterable)):
        img_size = (int(img_size),int(img_size))

    pts = cv_box-box_rect[0] if(centralize)else cv_box

    # Generate polygon mask, 2 for outline, 1 for inside pixels
    blmask = Image.new('L', (img_size[1],img_size[0]), 0)
    ImageDraw.Draw(blmask).polygon(pts.reshape(-1).tolist(), outline=2, fill=1)
    blmask = np.array(blmask)

    # use scipy.ndimage.morphology.distance_transform_edt to calculate distence
    # inside pixels is 1, outline and background is 0
    insides = np.where(blmask==1,1,0)
    # calculate distence with NEAREST BG pixel
    dst_mask = ndimage.distance_transform_edt(insides).astype(np.float32)
    # map value from [1,max distance] to [v_range,0]
    min_dst = 1.0 # min pixel distense
    # map to (0,maxv)
    dst_mask -= min_dst
    max_dst = np.max(dst_mask)
    if(max_dst==0.0):
        return (dst_mask+1.0)*0.5
    # map to (0,1)
    dst_mask /= max_dst
    # map to (1,0)
    dst_mask = 1.0-dst_mask
    # map to (v_range,0)
    dst_mask *= v_range
    # apply gaussian
    dst_mask = np.exp(-( (dst_mask)**2 / ( 2.0 * sigma**2 ) ) )
    dst_mask = np.where(blmask==1,dst_mask,0.0)
    
    return dst_mask

def cv_fill_by_box(img, src_box, dst_box, out_size):
    """
    Fill image from source box region to destination box region
    Args:
        img: (h,w) image
        src_box: (4,2) in (x,y) coordnate
        dst_box: (4,2) in (x,y) coordnate
        out_size: (new_h,new_w)
    Return:
        img: (new_h,new_w) image
    """
    src_box = src_box.astype(np.float32)
    dst_box = dst_box.astype(np.float32)
    det = src_box[0].copy()
    M = cv2.getPerspectiveTransform(src_box-det, dst_box-det)
    res = cv2.warpPerspective(img, M, (int(out_size[1]), int(out_size[0])))
    return res

def cv_create_affine_boxes(boxs,bx_wide:float = 0.8):
    """
    Calculate affine boxes
        boxs: (N,4,2) with (x,y)
        bx_wide: new_h = org_h * bx_wide
    Return:
        affin boxes: (N-1,4,2)
    """
    det = (1-min(1.0,max(0.3,bx_wide)))/2
    sp = boxs[:-1]
    top_center = (sp[:,1]+sp[:,0])/2.0
    bot_center = (sp[:,3]+sp[:,2])/2.0
    up_cent_sp = bot_center + (top_center-bot_center)*(1-det)
    dw_cent_sp = bot_center + (top_center-bot_center)*det
    ep = boxs[1:]
    top_center = (ep[:,1]+ep[:,0])/2.0
    bot_center = (ep[:,3]+ep[:,2])/2.0
    up_cent_ep = bot_center + (top_center-bot_center)*(1-det)
    dw_cent_ep = bot_center + (top_center-bot_center)*det

    return np.stack([up_cent_sp,up_cent_ep,dw_cent_ep,dw_cent_sp],axis=1)

def cv_rotate(image,angle,change_shape:bool=False):
    """
    Rotate image
    """
    h, w = image.shape[:-1]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if(change_shape):
        orgbox = np.array(((0,0),(w,0),(w,h),(0,h)))
        orgbox = np_polybox_rotate(orgbox,M)
        orgbox = np_polybox_minrect(orgbox)
        new_w,new_h = (orgbox[2]-orgbox[0])
        M[0,-1]+=(new_w-w)/2
        M[1,-1]+=(new_h-h)/2
        h, w = int(new_h), int(new_w)
    image = cv2.warpAffine(image, M, (w, h))
    return image, M

def cv_box_overlap(boxes1,boxes2):
    """
    Calculate box overlap in CV coordinate
    Args:
        boxes1: ((N1),2,4 or k) CV coordinate
        boxes1: ((N2),2,4 or k) CV coordinate
    Return:
        overlap: (N1,N2) array, 0.0 for non-overlap
    """
    if(len(boxes1.shape)==2):boxes1 = np.expand_dims(boxes1,0)
    if(len(boxes2.shape)==2):boxes2 = np.expand_dims(boxes2,0)
    ans = []
    for b1 in boxes1:
        tmp = []
        p1 = Polygon(b1)
        for b2 in boxes2:
            p2 = Polygon(b2)
            a = p1.intersection(p2).area
            tmp.append(a/(p1.area+p2.area-a) if(p1.intersects(p2))else 0.0)
        ans.append(tmp)
    return np.array(ans)

def cv_box_match(pred,gt,bg=None,ovth:float=0.5):
    """
    Box match algorithm
    Args:
        pred: ((N1),2,4) CV coordinate
        gt: ((N2),2,4) CV coordinate
        bg: optional, ((N3),2,4) CV coordinate
        ovth: overlap threshold, default is 0.5
    Return:
        id_list: id list of matched, non-negative for gt boxes, negative for bg boxes if have.
        Usage e.g. 
            if(id_list[i] and id_list[i]>=0):gt[id_list[i]]
            if(id_list[i] and id_list[i]<0):bg[id_list[i]]
    """
    pred = pred.reshape((-1,4,2))
    id_list = [None for i in range(pred.shape[0])]
    if(not isinstance(bg,type(None))):
        bg = bg.reshape((-1,4,2))
        ovs = cv_box_overlap(pred,bg)
        for i in range(pred.shape[0]):
            imax = np.argmax(ovs[i])
            if(ovs[i,imax]>=ovth):
                id_list[i] = -1-int(imax)

    ovs = cv_box_overlap(pred,gt)
    for i in range(pred.shape[0]):
        if(id_list[i]!=None):
            continue
        imax = np.argmax(ovs[i])
        if(ovs[i,imax]>=ovth):
            id_list[i] = imax
            
    return id_list

def cv_box2cvbox(boxes,image_size,box_format:str):
    """
    Convert regular box to CV2 coordinate.
    Args:
        boxes: (N,4 or 5) array
        image_size: (h,w)

    Return:
        (N,4,2) rectangle box in CV2 coordinate (polyxy).
        +------------> x
        | 
        | p0------p1
        | |        |
        | |        |
        | p3------p2
        y
    """
    if(box_format.lower()=='polyxy'):return boxes.reshape((-1,4,2))
    if(box_format.lower()=='polyyx'):
        boxes = boxes.reshape((-1,4,2))
        return np.stack([boxes[:,:,1],boxes[:,:,0]],axis=-1)
    if(not isinstance(boxes,np.ndarray)):boxes = np.array(boxes)
    if(len(boxes.shape)==1):boxes = np.expand_dims(boxes,0)
    boxes = np_box_transfrom(boxes[:,-4:],box_format,'xyxy')
    if(boxes.max()<=1.0):
        boxes*=(image_size[1],image_size[0],image_size[1],image_size[0],)
    
    return np.stack([
        boxes[:,0],boxes[:,1],#left-top
        boxes[:,2],boxes[:,1],#right-top
        boxes[:,2],boxes[:,3],#right-bottom
        boxes[:,0],boxes[:,3],#left-bottom
        ],axis=-1).reshape((-1,4,2))

def cv_cvbox2box(cv_4p_boxes,box_format:str):
    """
    Convvert CV box to box
    Args:
        cv_4p_boxes: (box_number,4,2) array 
        box_format: target box formot
    Return: (N,4) box array
    """
    minx = cv_4p_boxes[:,:,0].min(axis=-1)
    maxx = cv_4p_boxes[:,:,0].max(axis=-1)
    miny = cv_4p_boxes[:,:,1].min(axis=-1)
    maxy = cv_4p_boxes[:,:,1].max(axis=-1)

    return np_box_transfrom(np.stack([minx,miny,maxx,maxy],axis=-1),'xyxy',box_format)

def cv_crop_image_by_bbox(image, box, w_multi:int=None, h_multi:int=None,w_min:int=None, h_min:int=None):
    """
    Crop image by box.
    Args:
        image: numpy with shape (h,w,(3 or 1))
        box: shape (4,2) with ((x0,y0),(x1,y1),(x2,y2),(x3,y3))
        w_multi: final width will be k*w_multi
        h_multi: final height will be k*h_multi
        w_min: final width will greater than w_min
        h_min: final height will greater than h_min
    Return:
        subimg: (h,w,3)
        M: mapping matrix
    """
    if(not isinstance(box,np.ndarray)):box = np.array(box)
    w = (int)(np.linalg.norm(box[0] - box[1]))
    h = (int)(np.linalg.norm(box[0] - box[3]))
    if(w_min!=None):
        w = max(w,w_min)
    if(h_min!=None):
        h = max(h,h_min)
    width = max(1,w//int(w_multi))*int(w_multi) if(w_multi!=None and w_multi>0)else w
    height = max(1,h//int(h_multi))*int(h_multi) if(w_multi!=None and w_multi>0)else h
    # if h > w * 1.5:
    #     width = h
    #     height = w
    #     M = cv2.getPerspectiveTransform(np.float32(box),
    #                                     np.float32(np.array([[width, 0], [width, height], [0, height], [0, 0]])))
    # else:
    M = cv2.getPerspectiveTransform(np.float32(box),
                                    np.float32(np.array([[0, 0], [width, 0], [width, height], [0, height]])))

    warped = cv2.warpPerspective(image, M, (width, height))
    return warped, M

def cv_crop_image_by_polygon(image, box, w_multi:int=None, h_multi:int=None,w_min:int=None, h_min:int=None):
    """
    Crop image by polygon.
    Args:
        image: numpy with shape (h,w,3)
        box: shape (k,2) with (x,y)
        w_multi: final width will be k*w_multi
        h_multi: final height will be k*h_multi
        w_min: final width will greater than w_min
        h_min: final height will greater than h_min
    Return:
        subimg: (h,w,3)
        M: mapping matrix
    """
    box_rect = np_polybox_minrect(box)
    w,h = (box_rect[2]-box_rect[0]).astype(np.int16)
    sub_img,M = cv_crop_image_by_bbox(image, box_rect, w_multi, h_multi,w_min, h_min)
    pts = box-box_rect[0]
    pts = np_box_resize(pts,(h,w),sub_img.shape[:-1],'polyxy')
    # Generate polygon mask, 2 for outline, 1 for inside pixels
    blmask = Image.new('L', (sub_img.shape[1],sub_img.shape[0]), 0)
    ImageDraw.Draw(blmask).polygon(pts.reshape(-1).tolist(), outline=1, fill=1)
    blmask = np.array(blmask)
    blmask = np.expand_dims(blmask,-1)
    sub_img = np.where(blmask,sub_img,0)
    return sub_img,M

def cv_uncrop_image(sub_image, M, width:int, height:int):
    """
    UNDO cv_crop_image_by_bbox
    Args:
        sub_image: croped sub-image, numpy with shape (in_h,in_w,ch)
        M: map martex from cv_crop_image_by_bbox
        width: output image width
        height: output image height
    Return: image with shape (height,width,ch)
    """
    ret, IM = cv2.invert(M)
    return cv2.warpPerspective(sub_image, IM, (width, height))


def cv_get_box_from_mask(scoremap:np.ndarray, segmap:np.ndarray=None, score_th:float=0.5, seg_th:float=0.3):
    """
    Box detector with confidence map (and optional segmentation map).
    Args:
        scoremap: confidence map, ndarray in (h,w,1) or (h,w), range in [0,1], float
        segmap: segmentation map, ndarray in (h,w,1) or (h,w), range in [0,1], float
        score_th: threshold of score, float
        seg_th: threshold of segmentation, float
    Ret:
        detections: (N,4,2) box with polyxy format
        labels: map of labeled number
        mapper: list of number label for each box
    """
    img_h, img_w = scoremap.shape[0], scoremap.shape[1]
    scoremap = scoremap.reshape((img_h, img_w))
    if(isinstance(segmap,type(None))):
        segmap = np.zeros((img_h, img_w),dtype=scoremap.dtype)
    else:
        segmap = segmap.reshape((img_h, img_w))

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.logical_or(scoremap>=score_th,segmap>=seg_th).astype(np.uint8),
        connectivity=4)

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        rsize = stats[k, cv2.CC_STAT_AREA]
        if rsize < 10: continue

        # thresholding
        if np.max(scoremap[labels == k]) < score_th: continue

        # make segmentation map
        tmp = np.zeros(scoremap.shape, dtype=np.uint8)
        tmp[np.logical_and(labels == k,segmap<seg_th)] = 255

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(rsize * min(w, h) / (w * h)) * 2)
        x0 = max(0,x - niter)
        x1 = min(img_w,x + w + niter + 1)
        y0 = max(0,y - niter)
        y1 = min(img_h,y + h + niter + 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        tmp[y0:y1, x0:x1] = cv2.dilate(tmp[y0:y1, x0:x1], kernel)
        
        # make box
        np_contours = np.roll(np.array(np.where(tmp != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return np.array(det), labels, mapper

def cv_draw_poly(image,boxes,text=None,color = (0,255,0)):
    """
    Arg:
        img: ndarray in (h,w,c) in [0,255]
        boxes: ndarray, shape (boxes number,polygon point number,2 (x,y)) 
            or (polygon point number,2 (x,y))
    """
    image = image.astype(np.uint8)

    if(not isinstance(boxes,np.ndarray)):boxes = np.array(boxes)
    if(len(boxes.shape)==2):
        boxes=boxes.reshape((1,-1,2))
    if(isinstance(text,np.ndarray)):
        text = text.reshape((-1))        
    elif(not isinstance(text,type(None)) and isinstance(text,Iterable)):
        text = [text]*boxes.shape[0]

    for i in range(boxes.shape[0]):
        # the points is (polygon point number,1,2) in list
        cv2.polylines(image,[boxes[i].reshape((-1,1,2)).astype(np.int32)],True,color)
        if(not isinstance(None,type(None))):
            # print text box
            cv2.rectangle(image,
                (boxes[i,:,0].max(),max(0,boxes[i,:,1].min()-14)),#(x,y-10)
                (min(image.shape[1]-1,boxes[i,:,0].max()+10*len(str(text[i]))),boxes[i,:,1].min()),#(x+10N,y)
                (255, 0, 0),-1)
            # print text at top-right of box
            cv2.putText(
                image, text=str(text[i]), org=(boxes[i,:,0].max(),boxes[i,:,1].min()), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, lineType=cv2.LINE_AA, color=(255, 255, 255))

        
    return image

def cv_draw_rect(image,boxes,fm,text=None,color = (0,255,0)):
    """
    Draw rectangle box in image
    Arg:
        img: ndarray in (h,w,c)
        boxes: ndarray, shape (N,4)
        fm: box format in __DEF_FORMATS
    """
    image = image.astype(np.uint8)
    if(not isinstance(boxes,np.ndarray)):boxes = np.array(boxes)
    if(len(boxes.shape)==1):boxes = boxes.reshape((1,-1))
    fm = fm.lower() if(fm.lower() in __DEF_FORMATS)else __DEF_FORMATS[0]
    if(fm=='polyxy'):
        return cv_draw_poly(image,boxes.reshape((-1,2,4)),text)
    if(fm!='xyxy'):boxes = np_box_transfrom(boxes,fm,'xyxy')
    if(isinstance(text,np.ndarray)):
        text = text.reshape((-1))        
    elif(not isinstance(text,type(None)) and isinstance(text,Iterable)):
        text = [text]*boxes.shape[0]
    boxes = boxes.astype(np.int32)
    for i,o in enumerate(boxes):
        # top-left(x,y), bottom-right(x,y), color(r,b,g), thickness
        cv2.rectangle(image,(o[0],o[1]),(o[2],o[3]),color,3)
        if(isinstance(text,Iterable)):
            # print text box
            cv2.rectangle(image,
                (o[0],max(0,o[1]-14)),#(x,y-10)
                (min(image.shape[1]-1,o[0]+10*len(str(text[i]))),o[1]),#(x+10N,y)
                (255, 0, 0),-1)
            # print text at top-right of box
            cv2.putText(
                image, text=str(text[i]), org=(o[0],o[1]), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(255, 255, 255))
    return image

def cv_mask_image(image,mask,rate:float=0.7):
    """
    Mask single image.
    Args:
        image: (h1,w1,3 or 1) in np.int8
        mask: (h2,w2,(1)) np.float or (h2,w2,3) in np.int8
    Return:
        image: (h2,w2,3 or 1)
    """
    rate = min(max(0.1,rate),0.9)
    if(image.shape[-1]==1):
        image = np.broadcast_to(image,(image.shape[0],image.shape[1],3))
    if(len(mask.shape)==2 or mask.dtype!=np.uint8):
        mask = cv_heatmap(mask)
    
    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)
    if(image.shape!=mask.shape):
        image = cv2.resize(image,(mask.shape[1],mask.shape[0])).astype(np.uint8)

    return cv2.addWeighted(image,rate,mask,1.0-rate,0)

def cv_heatmap(img,clr = cv2.COLORMAP_JET):
    # clr demo see https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, clr)
    return img.astype(np.uint8)

def cv_watershed(org_img, mask, viz=False):
    """
    Watershed algorithm
    Args:
        org_img: original image, (h,w,chs) in np.uint8
        mask: mask image, (h,w) in np.float in [0,1]
    Return:

    """
    if(org_img.shape[0:2]!=mask.shape[0:2]):
        org_img = cv2.resize(org_img,(mask.shape[1],mask.shape[0]))
    org_img = org_img.astype(np.uint8)
    viz = lambda *args:None if(not viz)else cv2.imshow

    # apply threshold
    if(mask.max()>1.1):mask /= mask.max()
    mask*=255
    threshold = mask.max()*0.2
    b_mask = np.where(mask>=threshold,255,0).astype(np.uint8)
    viz("surface_bmask", b_mask)

    # apply filter
    kr = np.ones((3, 3), np.uint8)
    b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, kr, iterations=1)
    viz("surface_fg", b_mask)

    # find background
    sure_bg = cv2.dilate(b_mask, kr, iterations=3) 
    viz("sure_bg", sure_bg)

    # find foreground
    threshold = mask.max()*0.5
    sure_fg = np.where(mask>=threshold,255,0).astype(np.uint8)
    viz("sure_fg", sure_fg)

    unknown = sure_bg - sure_fg
    viz("unknown", unknown)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg,
                                                                         connectivity=4)
    
    markers = labels + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(org_img, markers=markers)
    org_img[markers == -1] = [0, 0, 255]
    viz("markers", markers)
    viz("org_img", org_img)

    boxes = []
    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)
    return np.array(boxes)

def cv_box_moving_vector(cv_src_box,cv_dst_box,image_size=None):
    """
    Calculate pixel level box moving vector
    Args:
        cv_src_box: sorce boxes (t-1) in cv coordinate, (N,3+,2)
        cv_dst_box: dest boxes (t) in cv coordinate, (N,3+,2)
            The shape of cv_src_box and cv_dst_box MUST be same
            and the box label is same in axis 0
        image_size: int/float or (h,w) or None

    Return:
        matrix_arr: (N,2,3) array of affine matrix
        matrix_map: affine matrix in pixel level, (h,w,6)
            or None if image_size is none
    """

    if(not isinstance(image_size,type(None)) and not isinstance(image_size,Iterable)):
        image_size = (image_size,image_size)
    matrix_map = np.zeros((image_size[0],image_size[1],6)) if(not isinstance(image_size,type(None)))else None
    assert(cv_src_box.shape==cv_dst_box.shape)
    matrix_list = []
    for i in range(cv_src_box.shape[0]):
        mm = cv2.getAffineTransform(cv_src_box[i,:3].astype(np.float32), cv_dst_box[i,:3].astype(np.float32))
        matrix_list.append(mm)
        if(not isinstance(image_size,type(None))):
            x1= int(max(cv_dst_box[i,:,0].min(),0))
            x2= int(min(cv_dst_box[i,:,0].max(),image_size[1]-1))
            y1= int(max(cv_dst_box[i,:,1].min(),0))
            y2= int(min(cv_dst_box[i,:,1].max(),image_size[0]-1))
            matrix_map[y1:y2,x1:x2]=mm.reshape((-1))

    return np.array(matrix_list),matrix_map

def cv_divd_polygon(box,n):
    """
    Divide polygon into N parts linearly
    v 0.5, need refine
    Args:
        box: (2k,2) x,y polygon points 
        n: n part
    Return:
        divplg: (n,4,2) x,y polygon points
    """
    # box: (2k,2) for polygon xy
    det_list = []
    total_det = 0
    ans = []
    # calculate center line length
    for j in range(box.shape[0]//2-1):
        up = box[j+1] - box[j]
        dw = box[-j-2] - box[-j-1]
        det = (math.sqrt(up[0]**2+up[1]**2)+math.sqrt(dw[0]**2+dw[1]**2))/2
        total_det+=det
        det_list.append(det)

    # single text lenth
    single_len = total_det/n
    last_length = 0.0
    last_ps = None
    for j in range(len(det_list)):
        # Skip small segment 
        if(det_list[j]+last_length<=single_len):
            last_length+=det_list[j]
            if(not isinstance(last_ps,type(None))):
                # Don't have last remnant
                last_ps = np.array((box[j],box[-j]))
            continue
        
        if(last_length>0.0):
            bx_len = det_list[j] + last_length
        else:
            bx_len = det_list[j]
        # total splited box number
        divs = int(bx_len/single_len)

        up_line = box[j+1] - box[j]
        dw_line = box[-j-2] - box[-j-1]
        up_sp,dw_sp = box[j],box[-j-1]
        sp_len = single_len - last_length
        up_last,dw_last = up_sp + up_line*sp_len/det_list[j],dw_sp + dw_line*sp_len/det_list[j]
        # add 1st box
        if(not isinstance(last_ps,type(None))):
            ans.append((last_ps[0],up_last,dw_last,last_ps[1]))
        else:
            ans.append((up_sp,up_last,dw_last,dw_sp))
        
        up_dxy = np.linspace(box[0],box[1],n+1)
        # add 2nd~end boxes
        for sub_bxi in range(divs-1):
            # append single polygon
            new_up_last,new_dw_last = up_last + up_line*single_len/det_list[j],dw_last + dw_line*single_len/det_list[j]
            ans.append((up_last,new_up_last,new_dw_last,dw_last))
            up_last,dw_last = new_up_last,new_dw_last

        # remain lenth for next last_length
        last_length = bx_len - single_len*divs
        last_ps = np.array((up_last,dw_last))
    if(last_length>=0.6*single_len):
        bxi = box.shape[0]//2-1
        ans.append((up_last,box[bxi],box[-bxi-1],dw_last))
    return np.stack(ans,axis=0)

def cv_gen_gaussian(cv_boxes,texts,img_size,ch_mask:np.ndarray=None,af_mask:np.ndarray=None,affin:bool=True):
    """
    Generate text level gaussian distribution
    Parameters:
        cv_boxes: ((box number),points number,2) in (x,y). Points number MUST be even
        texts: list with len((box number)) or str (for single box), set None to disable text level split
        img_size: tuple of (h,w)
        ch_mask: if not None, generator will add the gaussian distribution on mask
        af_mask: if not None, generator will add the gaussian distribution on mask
        affin: True to generate af_mask
    Return:
        ch_mask, af_mask: (h,w) mask on np.float32
        ch_boxes_list: list of character level box [(k,4,2) or None], same lenth with texts
        aff_boxes_list: list of character level affin box [(k-1,4,2) or None], same lenth with texts
    """
    if(len(cv_boxes.shape)==2):
        cv_boxes = np.expand_dims(cv_boxes,0)
    if(not isinstance(texts,list)):
        texts = [texts]
    if(not isinstance(img_size,Iterable)):
        img_size = (int(img_size),int(img_size))
    if(isinstance(ch_mask,type(None))):
        ch_mask = np.zeros(img_size,dtype=np.float32)
    elif(len(ch_mask.shape)==3):
        ch_mask = ch_mask[:,:,0]
    if(affin):
        if(isinstance(af_mask,type(None))):
            af_mask = np.zeros(img_size,dtype=np.float32)
        elif(len(af_mask.shape)==3):
            af_mask = af_mask[:,:,0]
    cv_boxes = cv_boxes.copy()
    ch_boxes_list = []
    aff_boxes_list = []

    for bxi in range(cv_boxes.shape[0]):
        box = cv_boxes[bxi]
        if(texts):
            txt = texts[bxi].strip()
            total_len = len(txt)
        else:
            # diasble split
            total_len = -1
        if(total_len==1):
            total_len = -1
        ch_boxes = [] # (k,4,2)
        # Generate character level boxes
        if(total_len<0):
            # diasble split
            if(box.shape[0]==4):
                ch_boxes.append(box)
            else:
                # polygon
                # for j in range(box.shape[0]//2-1):
                #     ch_boxes.append((box[j],box[j+1],box[-j-2],box[-j-1]))
                ch_boxes.append((box[0],box[box.shape[0]//2-1],box[-box.shape[0]//2],box[-1]))
        elif(box.shape[0]==4):
            # rectangle
            up_dxy = np.linspace(box[0],box[1],total_len+1)
            dw_dxy = np.linspace(box[3],box[2],total_len+1)
            for txi in range(total_len):
                if(txt[txi] in ' '):
                    continue
                elif(txt[txi] in '._,'):
                    # 
                    ch_boxes.append(((up_dxy[txi]+dw_dxy[txi])/2,(up_dxy[txi+1]+dw_dxy[txi+1])/2,dw_dxy[txi+1],dw_dxy[txi]))
                else:
                    ch_boxes.append((up_dxy[txi],up_dxy[txi+1],dw_dxy[txi+1],dw_dxy[txi]))
        else:
            # box: (2k,2) for polygon xy
            det_list = []
            total_det = 0
            # calculate center line length
            for j in range(box.shape[0]//2-1):
                up = box[j+1] - box[j]
                dw = box[-j-2] - box[-j-1]
                det = (math.sqrt(up[0]**2+up[1]**2)+math.sqrt(dw[0]**2+dw[1]**2))/2
                total_det+=det
                det_list.append(det)

            # single text lenth
            single_len = total_det/total_len
            txi = 0
            last_length = 0.0
            last_ps = None
            for j in range(len(det_list)):
                # Skip small segment 
                if(det_list[j]+last_length<=single_len):
                    last_length+=det_list[j]
                    if(not isinstance(last_ps,type(None))):
                        # Don't have last remnant
                        last_ps = np.array((box[j],box[-j]))
                    continue
                
                if(last_length>0.0):
                    bx_len = det_list[j] + last_length
                else:
                    bx_len = det_list[j]
                # total splited box number
                divs = int(bx_len/single_len)

                up_line = box[j+1] - box[j]
                dw_line = box[-j-2] - box[-j-1]
                up_sp,dw_sp = box[j],box[-j-1]
                sp_len = single_len - last_length
                up_last,dw_last = up_sp + up_line*sp_len/det_list[j],dw_sp + dw_line*sp_len/det_list[j]
                # add 1st box
                if(not isinstance(last_ps,type(None))):
                    ch_boxes.append((last_ps[0],up_last,dw_last,last_ps[1]))
                else:
                    ch_boxes.append((up_sp,up_last,dw_last,dw_sp))
                
                up_dxy = np.linspace(box[0],box[1],total_len+1)
                # add 2nd~end boxes
                for sub_bxi in range(divs-1):
                    # append single polygon
                    new_up_last,new_dw_last = up_last + up_line*single_len/det_list[j],dw_last + dw_line*single_len/det_list[j]
                    ch_boxes.append((up_last,new_up_last,new_dw_last,dw_last))
                    up_last,dw_last = new_up_last,new_dw_last

                # remain lenth for next last_length
                last_length = bx_len - single_len*divs
                last_ps = np.array((up_last,dw_last))
            if(last_length>=0.6*single_len):
                bxi = box.shape[0]//2-1
                ch_boxes.append((up_last,box[bxi],box[-bxi-1],dw_last))
        # draw boxes from ch_boxes
        ch_boxes = np.array(ch_boxes) # (k,4,2)
        if(ch_boxes.shape[0]==0):
            ch_boxes_list.append(None)
            aff_boxes_list.append(None)
            continue
        ch_boxes_list.append(ch_boxes)

        # draw gaussian mask
        ch_boxes_rec = np_polybox_minrect(ch_boxes,'polyxy')
        dxy = ch_boxes_rec[:,2]-ch_boxes_rec[:,0]
        # Generate chmask
        for chi in range(ch_boxes.shape[0]):
            deta_x, deta_y = dxy[chi].astype(np.int16)
            if(deta_x*deta_y<=0):
                continue
            min_x = max(int(ch_boxes_rec[chi,0,0]),0)
            min_y = max(int(ch_boxes_rec[chi,0,1]),0)

            # gaussian = cv_gaussian_kernel_2d(kernel_size=(deta_y, deta_x))
            gaussian = np_2d_gaussian((deta_y, deta_x))
            res = cv_fill_by_box(gaussian, ch_boxes_rec[chi], ch_boxes[chi],(deta_y,deta_x))
            max_v = np.max(res)
            if(max_v > 0):
                # res/=max_v
                sub_mask = ch_mask[min_y:min_y+res.shape[0],min_x:min_x+res.shape[1]]
                sub_mask = np.where(sub_mask>res[0:sub_mask.shape[0],0:sub_mask.shape[1]],sub_mask,res[0:sub_mask.shape[0],0:sub_mask.shape[1]])
                ch_mask[min_y:min_y+res.shape[0],min_x:min_x+res.shape[1]] = sub_mask
        # Affin mask
        if(not affin):
            continue
        if(ch_boxes.shape[0]<=1):
            aff_boxes_list.append(None)
            continue
        aff_boxes = cv_create_affine_boxes(ch_boxes)
        aff_boxes_list.append(aff_boxes)
        aff_boxes_rec = np_polybox_minrect(aff_boxes,'polyxy')
        dxy = aff_boxes_rec[:,2]-aff_boxes_rec[:,0]
        for afi in range(aff_boxes.shape[0]):
            deta_x, deta_y = dxy[afi].astype(np.int16)
            if(deta_x*deta_y<=0):
                continue
            min_x = max(int(aff_boxes_rec[afi,0,0]),0)
            min_y = max(int(aff_boxes_rec[afi,0,1]),0)

            # gaussian = cv_gaussian_kernel_2d(kernel_size=(deta_y, deta_x))
            gaussian = np_2d_gaussian((deta_y, deta_x))
            res = cv_fill_by_box(gaussian, aff_boxes_rec[afi], aff_boxes[afi],(deta_y, deta_x))
            max_v = np.max(res)
            if(max_v > 0):
                # res/=max_v
                sub_mask = af_mask[min_y:min_y+res.shape[0],min_x:min_x+res.shape[1]]
                sub_mask = np.where(sub_mask>res,sub_mask,res)
                af_mask[min_y:min_y+res.shape[0],min_x:min_x+res.shape[1]] = sub_mask
    
    return ch_mask,af_mask,ch_boxes_list,aff_boxes_list

def cv_watershed_gen_gaussian(img,cv_boxes,texts,img_size,ch_mask:np.ndarray=None,af_mask:np.ndarray=None,affin:bool=True):
    """
    Generate text level gaussian distribution use watershed
    if watershed can't generate correct text region, use cv_gen_gaussian
    Parameters:
        img: orginal image in np.uint8
        cv_boxes: ((box number),points number,2) in (x,y). Points number MUST be even
        texts: list with len((box number)) or str (for single box), set None to disable text level split
        img_size: tuple of (h,w)
        ch_mask: if not None, generator will add the gaussian distribution on mask
        af_mask: if not None, generator will add the gaussian distribution on mask
        affin: True to generate af_mask
    Return:
        ch_mask, af_mask: (h,w) mask on np.float32
        ch_boxes_list: list of character level box [(k,4,2) or None], same lenth with texts
        aff_boxes_list: list of character level affin box [(k-1,4,2) or None], same lenth with texts
    """
    # Check parameters
    if(isinstance(ch_mask,type(None))):
        ch_mask = np.zeros(img_size,dtype=np.float32)
    if(isinstance(af_mask,type(None))):
        af_mask = np.zeros(img_size,dtype=np.float32)
    if(not isinstance(texts,list)):
        texts = [texts]
    if(len(cv_boxes.shape)==2):
        cv_boxes = np.expand_dims(cv_boxes,0)
    if(img.dtype!=np.uint8):
        img = img.astype(np.uint8)

    for bxi in range(cv_boxes.shape[0]):
        targ_len = len(texts[bxi])
        sub_x = cv_crop_image_by_polygon(img,cv_boxes[bxi])
        sub_gray = cv2.cvtColor(sub_x,cv2.COLOR_BGR2GRAY)
        ret,sub_bin = cv2.threshold(sub_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(sub_bin,connectivity=8)
        # if(nLabels<targ_len):
        #     # faild to saperate text
        #     cv_gen_gaussian()
        

# 
# ===============================================
# ==================== Class ====================
# ===============================================
#

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
                sample['box'] = [np_box_rescale(o,rate,fmt) if(o[:,-4:].max()>1.0)else o for o in box]
            elif(box[:,-4:].max()>1.0):
                sample['box'] = np_box_rescale(box,rate,fmt)
        if('gtmask' in sample):
            sample['gtmask'] = sample['gtmask'].resize(s_shape)

        return sample

