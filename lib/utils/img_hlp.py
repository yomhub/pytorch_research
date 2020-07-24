from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

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
    CV format 4 points: (x,y) in 
    +------------> x
    | 
    | p2------p3
    | |        |
    | |        |
    | p0------p1
    y
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

def np_box_to_points(boxes:np.ndarray,img_size=None,box_format:str='xywh'):
    """
    Convert box to 4 points.
    Args:
        boxes: (N,4) np.ndarray
        img_size: tuple, (y,x) or int for both yx, 
            if img_size==None AND boxes is normalized,
            return points coordinate in normalized.
        box_format in ['yxyx','xyxy','xywh','cxywh']
    Return:
        (N,4,2) np.ndarray
    """
    if(img_size!=None and not(isinstance(img_size,list) or isinstance(img_size,tuple))):
        img_size = (img_size,img_size)
    if(box_format not in __DEF_FORMATS):box_format = __DEF_FORMATS[0]
    boxes = np_box_transfrom(boxes,box_format,'xyxy')
    ret = []
    if(img_size!=None and boxes.max()<=1.0):
        for o in boxes:
            x1=o[0]*img_size[1]
            y1=o[1]*img_size[0]
            x2=o[2]*img_size[1]
            y2=o[3]*img_size[0]
            ret.append([(x1,y2),(x2,y2),(x1,y1),(x2,y1)])
    else:
        for o in boxes:
            ret.append([(o[0],o[3]),(o[2],o[3]),(o[0],o[1]),(o[2],o[1])])
    return np.array(ret,dtype=boxes.dtype)

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

def np_2d_gaussian(img_size,x_range=(-1.0,1.0),y_range=(-1.0,1.0),sigma:float=1.0,mu:float=0.0):
    """
    Generate gaussian distribution.
    Args: 
        x_range/y_range: tuple, (a,b) or float for (-a,a), 
        sigma/mu: float
        img_size: tuple, (y,x) or int for both yx, 
    Return 2d gaussian distribution in numpy.
    """
    if(not(isinstance(img_size,list) or isinstance(img_size,tuple))):
        img_size = (img_size,img_size)
    if(not(isinstance(x_range,list) or isinstance(x_range,tuple))):
        x_range = (-x_range,x_range)
    if(not(isinstance(y_range,list) or isinstance(y_range,tuple))):
        y_range = (-y_range,y_range)
    dx, dy = np.meshgrid(
        np.linspace(x_range[0],x_range[1],img_size[1]), 
        np.linspace(y_range[0],y_range[1],img_size[0]))

    d = np.sqrt(dx**2+dy**2)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

# def np_gen_gaussian_in_img(img_size,boxes:np.ndarray,threshold=None):
#     """
#     Generate gaussian distribution in image by box.
#     Args: 
#         img_size: tuple, (y,x) or int for both yx, 
#         boxes: (N,4) np.ndarray in (cx,cy,w,h)
#     Return 2d gaussian distribution in numpy.
#     """
#     if(not(isinstance(img_size,list) or isinstance(img_size,tuple))):
#         img_size = (img_size,img_size)
#     if(boxes.max()>1.0):
#         boxes = np_box_nor(boxes,img_size,'cxywh')
#     gaussian_size = (int(boxes[:,-1].max()*img_size[0]),int(boxes[:,-2].max()*img_size[1]))
#     gaussian_map = np_2d_gaussian(gaussian_size)
#     if(threshold!=None):
#         gaussian_map = np.where(gaussian_map>=float(threshold),gaussian_map,0.0)
#     img = np.zeros(img_size)
#     for box in boxes:
#         box[]

#     return img

def cv_score2boxs(score:np.ndarray,threshold=None):
    """
    Convert score 2 box
    ???
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
                sample['box'] = [np_box_rescale(o,rate,fmt) if(o[:,-4:].max()>1.0)else o for o in box]
            elif(box[:,-4:].max()>1.0):
                sample['box'] = np_box_rescale(box,rate,fmt)
        if('gtmask' in sample):
            sample['gtmask'] = sample['gtmask'].resize(s_shape)

        return sample

class GaussianTransformer(object):

    def __init__(self, imgSize=512, region_threshold=0.4,
                 affinity_threshold=0.2):
        distanceRatio = 3.34
        self.region_threshold = region_threshold
        self.imgSize = imgSize
        self.standardGaussianHeat = self._gen_gaussian_heatmap(imgSize, distanceRatio)

        np_contours = np.roll(np.array(np.where(self.standardGaussianHeat >= region_threshold * 255)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.regionbox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)

        np_contours = np.roll(np.array(np.where(self.standardGaussianHeat >= affinity_threshold * 255)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.affinitybox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)

        self.oribox = np.array([[0, 0, 1], [imgSize - 1, 0, 1], [imgSize - 1, imgSize - 1, 1], [0, imgSize - 1, 1]],
                               dtype=np.int32)

    def _gen_gaussian_heatmap(self, imgSize, distanceRatio):
        gf = lambda x: np.exp(-(1 / 2) * (x ** 2))
        lx = ly = imgSize
        dx, dy = np.meshgrid(np.arange(lx), np.arange(ly))
        dx -= lx//2
        dy -= ly//2
        d = np.sqrt(dx**2+dy**2)*distanceRatio/max(ly//2,lx//2)
        return np.clip(gf(d) * 255, 0, 255)#.astype(np.uint8)

    def _test(self):
        sigma = 10
        spread = 3
        extent = int(spread * sigma)
        center = spread * sigma / 2
        gaussian_heatmap = np.zeros([extent, extent], dtype=np.float32)

        for i_ in range(extent):
            for j_ in range(extent):
                gaussian_heatmap[i_, j_] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                    -1 / 2 * ((i_ - center - 0.5) ** 2 + (j_ - center - 0.5) ** 2) / (sigma ** 2))

        gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255).astype(np.uint8)
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        threshhold_guassian = cv2.applyColorMap(gaussian_heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(images_folder, 'test_guassian.jpg'), threshhold_guassian)

    def add_region_character(self, image, target_bbox, regionbox=None):

        if np.any(target_bbox < 0) or np.any(target_bbox[:, 0] > image.shape[1]) or np.any(
                target_bbox[:, 1] > image.shape[0]):
            return image
        affi = False
        if regionbox is None:
            regionbox = self.regionbox.copy()
        else:
            affi = True

        M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(target_bbox))
        oribox = np.array(
            [[[0, 0], [self.imgSize - 1, 0], [self.imgSize - 1, self.imgSize - 1], [0, self.imgSize - 1]]],
            dtype=np.float32)
        test1 = cv2.perspectiveTransform(np.array([regionbox], np.float32), M)[0]
        real_target_box = cv2.perspectiveTransform(oribox, M)[0]
        real_target_box = np.int32(real_target_box)
        real_target_box[:, 0] = np.clip(real_target_box[:, 0], 0, image.shape[1])
        real_target_box[:, 1] = np.clip(real_target_box[:, 1], 0, image.shape[0])

        if np.any(target_bbox[0] < real_target_box[0]) or (
                target_bbox[3, 0] < real_target_box[3, 0] or target_bbox[3, 1] > real_target_box[3, 1]) or (
                target_bbox[1, 0] > real_target_box[1, 0] or target_bbox[1, 1] < real_target_box[1, 1]) or (
                target_bbox[2, 0] > real_target_box[2, 0] or target_bbox[2, 1] > real_target_box[2, 1]):
            # if False:
            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
            warped = np.array(warped, np.uint8)
            image = np.where(warped > image, warped, image)
        else:
            xmin = real_target_box[:, 0].min()
            xmax = real_target_box[:, 0].max()
            ymin = real_target_box[:, 1].min()
            ymax = real_target_box[:, 1].max()

            width = xmax - xmin
            height = ymax - ymin
            _target_box = target_bbox.copy()
            _target_box[:, 0] -= xmin
            _target_box[:, 1] -= ymin
            _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
            warped = np.array(warped, np.uint8)
            if warped.shape[0] != (ymax - ymin) or warped.shape[1] != (xmax - xmin):
                print("region (%d:%d,%d:%d) warped shape (%d,%d)" % (
                    ymin, ymax, xmin, xmax, warped.shape[1], warped.shape[0]))
                return image
            image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
                                                   image[ymin:ymax, xmin:xmax])
        return image

    def add_affinity_character(self, image, target_bbox):
        return self.add_region_character(image, target_bbox, self.affinitybox)

    def add_affinity(self, image, bbox_1, bbox_2):
        center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
        bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
        tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
        br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

        affinity = np.array([tl, tr, br, bl])

        return self.add_affinity_character(image, affinity.copy()), np.expand_dims(affinity, axis=0)

    def generate_region(self, image_size, bboxes):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        for i in range(len(bboxes)):
            character_bbox = np.array(bboxes[i].copy())
            for j in range(bboxes[i].shape[0]):
                target = self.add_region_character(target, character_bbox[j])

        return target

    def generate_affinity(self, image_size, bboxes, words):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        affinities = []
        for i in range(len(words)):
            character_bbox = np.array(bboxes[i])
            total_letters = 0
            for char_num in range(character_bbox.shape[0] - 1):
                target, affinity = self.add_affinity(target, character_bbox[total_letters],
                                                     character_bbox[total_letters + 1])
                affinities.append(affinity)
                total_letters += 1
        if len(affinities) > 0:
            affinities = np.concatenate(affinities, axis=0)
        return target, affinities

    def saveGaussianHeat(self,folder_name:str='gaussian_img'):
        images_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)),folder_name)
        cv2.imwrite(os.path.join(images_folder, 'standard.jpg'), self.standardGaussianHeat)
        warped_color = cv2.applyColorMap(self.standardGaussianHeat.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.polylines(warped_color, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'standard_color.jpg'), warped_color)
        standardGaussianHeat1 = self.standardGaussianHeat.copy()
        threshhold = self.region_threshold * 255
        standardGaussianHeat1[standardGaussianHeat1 > 0] = 255
        threshhold_guassian = cv2.applyColorMap(standardGaussianHeat1.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.polylines(threshhold_guassian, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'threshhold_guassian.jpg'), threshhold_guassian)


if __name__ == '__main__':
    gaussian = GaussianTransformer(512, 0.4, 0.2)
    gaussian.saveGaussianHeat()
    gaussian._test()
    bbox0 = np.array([[[0, 0], [100, 0], [100, 100], [0, 100]]])
    image = np.zeros((500, 500), np.uint8)
    # image = gaussian.add_region_character(image, bbox)
    bbox1 = np.array([[[100, 0], [200, 0], [200, 100], [100, 100]]])
    bbox2 = np.array([[[100, 100], [200, 100], [200, 200], [100, 200]]])
    bbox3 = np.array([[[0, 100], [100, 100], [100, 200], [0, 200]]])

    bbox4 = np.array([[[96, 0], [151, 9], [139, 64], [83, 58]]])
    # image = gaussian.add_region_character(image, bbox)
    # print(image.max())
    image = gaussian.generate_region((500, 500, 1), [bbox4])
    target_gaussian_heatmap_color = (np.clip(image.copy() / 255, 0, 1) * 255).astype(np.uint8)
    target_gaussian_heatmap_color = cv2.applyColorMap(target_gaussian_heatmap_color, cv2.COLORMAP_JET)
    cv2.imshow("test", target_gaussian_heatmap_color)
    cv2.imwrite("test.jpg", target_gaussian_heatmap_color)
    cv2.waitKey()
    # weight, target = gaussian.generate_target((1024, 1024, 3), bbox.copy())
    # target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(weight.copy() / 255)
    # cv2.imshow('test', target_gaussian_heatmap_color)
    # cv2.waitKey()
    # cv2.imwrite("test.jpg", target_gaussian_heatmap_color)