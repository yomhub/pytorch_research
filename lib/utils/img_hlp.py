from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
import math
from collections import Iterable
import Polygon as plg
from shapely.geometry import Polygon
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

def np_box_transfrom(box:np.ndarray,src_format:str,dst_format:str)->np.ndarray:
    """
    Box transfrom in ['yxyx','xyxy','xywh','cxywh']
    """
    src_format = src_format.lower()
    dst_format = dst_format.lower()
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

def np_corp_points(points:np.ndarray,ret_cod_len:int=4):
    """
    Corp 4 points to rectangle box
    Args:
        points: (N,4,2) numpy with (x,y) coordinate
    Return:
        boxes: (N,4,2) numpy with (x,y) with ret_cod_len=2
        boxes: (N,4) numpy with (x1,y1,x2,y2) with ret_cod_len=4
    """
    points = points.copy()
    minx = np.min(points[:,:,0],keepdims=0)
    miny = np.min(points[:,:,1],keepdims=0)
    maxx = np.max(points[:,:,0],keepdims=0)
    maxy = np.max(points[:,:,1],keepdims=0)
    return np.array([minx,miny,maxx,maxy]) if(ret_cod_len==4)else \
        np.array([[minx,miny],[maxx,miny],[maxx,maxy],[minx,maxy]])

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

def np_2d_gaussian(img_size,x_range=(-1.0,1.0),y_range=(-1.0,1.0),sigma:float=1.0,mu:float=0.0):
    """
    Generate gaussian distribution.
    Args: 
        x_range/y_range: tuple, (a,b) or float for (-a,a), 
        sigma/mu: float
        img_size: tuple, (y,x) or int for both yx, 
    Return 2d gaussian distribution in numpy.
    """
    if(not isinstance(img_size,Iterable)):
        img_size = (img_size,img_size)
    if(not isinstance(x_range,Iterable)):
        x_range = (-x_range,x_range)
    if(not isinstance(y_range,Iterable)):
        y_range = (-y_range,y_range)
    dx, dy = np.meshgrid(
        np.linspace(x_range[0],x_range[1],img_size[1]), 
        np.linspace(y_range[0],y_range[1],img_size[0]))

    d = np.sqrt(dx**2+dy**2)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

def cv_box_overlap(boxes1,boxes2):
    """
    Calculate box overlap in CV coordinate
    Args:
        boxes1: ((N1),2,4) CV coordinate
        boxes1: ((N2),2,4) CV coordinate
    Return:
        overlap: (N1,N2) array, -1 for non-overlap
    """
    if(len(boxes1.shape)==2):boxes1 = np.expand_dims(boxes1,0)
    if(len(boxes2.shape)==2):boxes2 = np.expand_dims(boxes2,0)
    ans = []
    for b1 in boxes1:
        tmp = []
        p1 = Polygon(b1)
        for b2 in boxes2:
            p2 = Polygon(b2)
            tmp.append(p1.intersection(p2).area if(p1.intersects(p2))else -1.0)
        ans.append(tmp)
    return np.array(ans)

def cv_box2cvbox(boxes,image_size,box_format:str):
    """
    Convert regular box to CV2 coordinate.
    Args:
        boxes: (N,4 or 5) array
        image_size: (h,w)

    Return:
        (N,4,2) rectangle box in CV2 coordinate.
        +------------> x
        | 
        | p0------p1
        | |        |
        | |        |
        | p3------p2
        y
    """
    if(box_format.lower()=='polyxy'):return boxes
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

def cv_crop_image_by_bbox(image, box, w_multi:int=None, h_multi:int=None):
    """
    Crop image by box, using cv2.
    Args:
        image: numpy with shape (h,w,3)
        box: shape (4,2) with ((x0,y0),(x1,y1),(x2,y2),(x3,y3))
        w_multi: final width will be k*w_multi
        h_multi: final height will be k*h_multi

    """
    if(not isinstance(box,np.ndarray)):box = np.array(box)
    w = (int)(np.linalg.norm(box[0] - box[1]))
    h = (int)(np.linalg.norm(box[0] - box[3]))
    width = max(1,w//int(w_multi))*int(w_multi) if(w_multi!=None and w_multi>0)else w
    height = max(1,h//int(h_multi))*int(h_multi) if(w_multi!=None and w_multi>0)else h
    if h > w * 1.5:
        width = h
        height = w
        M = cv2.getPerspectiveTransform(np.float32(box),
                                        np.float32(np.array([[width, 0], [width, height], [0, height], [0, 0]])))
    else:
        M = cv2.getPerspectiveTransform(np.float32(box),
                                        np.float32(np.array([[0, 0], [width, 0], [width, height], [0, height]])))

    warped = cv2.warpPerspective(image, M, (width, height))
    return warped, M

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

def cv_getDetCharBoxes_core(scoremap:np.ndarray, segmap:np.ndarray=None, score_th:float=0.5, seg_th:float=0.4):
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
    if(len(boxes)==2):
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

def cv_heatmap(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

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
        cv_src_box: sorce boxes (t-1) in cv coordinate, (N,4,2)
        cv_dst_box: dest boxes (t) in cv coordinate, (N,4,2)
            The shape of cv_src_box and cv_dst_box MUST be same
            and the box label is same in axis 0
        image_size: int/float or (h,w) or None

    Return:
        matrix_list: list of affine matrix
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
        image = img
        img_size = image.shape[-3:-1]
        min_rate = np.divide(self.min_size,img_size)
        max_rate = np.divide(self.max_size,img_size) if(self.max_size!=None)else np.maximum(min_rate,(1.5,1.5))
        rate = np.random.uniform(min_rate,max_rate,2)
        img_size *= rate

        s_shape = (image.shape[0],int(img_size[0]),int(img_size[1]),image.shape[-1]) if(len(image.shape)==4)else (int(img_size[0]),int(img_size[1]),image.shape[-1])
        img = np.resize(image,s_shape)

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

class GaussianTransformer(object):
    """
    Text level gaussian generator.
    Args:
        img_size: int or tuple for (y,x)

    """
    def __init__(self, img_size=512, region_threshold=0.4,
                 affinity_threshold=0.2):
        
        self.region_threshold = region_threshold
        img_size = img_size if(isinstance(img_size,Iterable))else (img_size,img_size)

        self.standardGaussianHeat = np_2d_gaussian(img_size)
        
        np_contours = np.roll(np.array(np.where(self.standardGaussianHeat >= region_threshold)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.regionbox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

        np_contours = np.roll(np.array(np.where(self.standardGaussianHeat >= affinity_threshold)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.affinitybox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        # CV coordinate (x,y,1)
        self.oribox = np.array([[0, 0, 1], [img_size[1] - 1, 0, 1], [img_size[1] - 1, img_size[0] - 1, 1], [0, img_size[0] - 1, 1]],
                               dtype=np.float32)

    def add_region_character(self, mask, cv_4p_box, regionbox=None):
        """
        Args:
            mask: ndarray with shape (height, width) and value in [0,1]
            cv_4p_box: ndarray with shape (4,2) with (x,y)
        """
        height, width = mask.shape[:2]
        np.clip(cv_4p_box[:,0],0,width-1,out=cv_4p_box[:,0])
        np.clip(cv_4p_box[:,1],0,height-1,out=cv_4p_box[:,1])

        if(isinstance(regionbox,type(None))):
            regionbox = self.regionbox

        M = cv2.getPerspectiveTransform(regionbox, cv_4p_box)
        real_target_box = cv2.perspectiveTransform(np.expand_dims(self.oribox[:,:2],0), M)[0]
        np.clip(real_target_box[:, 0], 0, width-1, out=real_target_box[:,0])
        np.clip(real_target_box[:, 1], 0, height-1, out=real_target_box[:,1])

        if np.any(cv_4p_box[0] < real_target_box[0]) or (
                cv_4p_box[3, 0] < real_target_box[3, 0] or cv_4p_box[3, 1] > real_target_box[3, 1]) or (
                cv_4p_box[1, 0] > real_target_box[1, 0] or cv_4p_box[1, 1] < real_target_box[1, 1]) or (
                cv_4p_box[2, 0] > real_target_box[2, 0] or cv_4p_box[2, 1] > real_target_box[2, 1]):
            # if False:
            warped = cv2.warpPerspective(self.standardGaussianHeat, M, (width, height))
            mask = np.where(warped > mask, warped, mask)
        else:
            xmin = real_target_box[:, 0].min()
            xmax = real_target_box[:, 0].max()
            ymin = real_target_box[:, 1].min()
            ymax = real_target_box[:, 1].max()

            width = xmax - xmin
            height = ymax - ymin
            _target_box = cv_4p_box.copy()
            _target_box[:, 0] -= xmin
            _target_box[:, 1] -= ymin
            _M = cv2.getPerspectiveTransform(regionbox, _target_box)
            warped = cv2.warpPerspective(self.standardGaussianHeat, _M, (width, height))

            if warped.shape[0] != (ymax - ymin) or warped.shape[1] != (xmax - xmin):
                print("region (%d:%d,%d:%d) warped shape (%d,%d)" % (
                    ymin, ymax, xmin, xmax, warped.shape[1], warped.shape[0]))
                return mask
            ymin,ymax,xmin,xmax = int(ymin),int(ymax),int(xmin),int(xmax)
            mask[ymin:ymax, xmin:xmax] = np.where(warped > mask[ymin:ymax, xmin:xmax], warped,
                                                   mask[ymin:ymax, xmin:xmax])
        return mask

    def generate_region(self, image_size, cv_4p_boxes_list:list, mask:np.ndarray=None):
        """
        Generate word region mask.
        Args:
            image_size: int or tuple (y,x)
            cv_4p_boxes_list: LIST of N connected character box (N,4,2) with (x,y)
            mask: None or float ndarray with shape (image_size) or (image_size,1)
        Return: mask with value in [0,1]
        """
        if(not isinstance(image_size,Iterable)):image_size = (image_size,image_size)
        if(isinstance(mask,type(None))):
            mask = np.zeros(image_size, dtype=np.float32)
        elif(len(mask.shape)>2):
            mask = mask.reshape(image_size)

        for word in cv_4p_boxes_list:
            word = word.astype(np.float32)
            for box in word:
                mask = self.add_region_character(mask, box)

        return mask

    def generate_affinity(self, image_size, cv_4p_boxes_list:list, mask:np.ndarray=None):
        """
        Generate word affinity.
        Args:
            image_size: int or tuple (y,x)
            cv_4p_boxes_list: LIST of N connected character box (N,4,2) with (x,y)
            mask: None or float ndarray with shape (image_size) or (image_size,1)
        Return: mask with value in [0,1]
        """

        if(not isinstance(image_size,Iterable)):image_size = (image_size,image_size)
        if(isinstance(mask,type(None))):
            mask = np.zeros(image_size, dtype=np.float32)
        elif(len(mask.shape)>2):
            mask = mask.reshape(image_size)

        for word in cv_4p_boxes_list:
            word = word.astype(np.float32)
            if(word.shape[0]<=1):continue
            src,dst = word[:-1],word[1:]
            center_src, center_dst = np.mean(src, axis=1), np.mean(dst, axis=1)
            bboxes = np.stack([
                np.mean([src[:,0,:], src[:,1,:], center_src], axis=0), # top-left 
                np.mean([dst[:,0,:], dst[:,1,:], center_dst], axis=0), # top-right
                np.mean([dst[:,2,:], dst[:,3,:], center_dst], axis=0), # bottom-right
                np.mean([src[:,2,:], src[:,3,:], center_src], axis=0), # bottom-left 
                ],axis = 1)
            for box in bboxes:
                mask = self.add_region_character(mask, box, self.affinitybox)

        return mask

    def split_word_box(self,cv_4p_boxes,split_num):
        """
        Args:
            cv_4p_boxes: word box (4,2) with (x,y)
            split_num: int
        Return: N connected character box (N,4,2) with (x,y)
        """
        if(split_num<=1):return np.expand_dims(cv_4p_boxes,0)
        # w = (np.sum(np.sqrt(np.square(cv_4p_boxes[1]-cv_4p_boxes[0])))+np.sum(np.sqrt(np.square(cv_4p_boxes[2]-cv_4p_boxes[3]))))/2
        dw0 = (cv_4p_boxes[1]-cv_4p_boxes[0])/split_num
        dw1 = (cv_4p_boxes[2]-cv_4p_boxes[3])/split_num
        # h = (np.sum(np.sqrt(np.square(cv_4p_boxes[1]-cv_4p_boxes[0])))+np.sum(np.sqrt(np.square(cv_4p_boxes[2]-cv_4p_boxes[3]))))/2

        ret = []
        for i in range(split_num):
            ret.append([cv_4p_boxes[0]+i*dw0,
            cv_4p_boxes[0]+(i+1)*dw0,
            cv_4p_boxes[3]+i*dw1,
            cv_4p_boxes[3]+(i+1)*dw1,])

        return np.array(ret,dtype=cv_4p_boxes.dtype)

