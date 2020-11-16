import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from matplotlib import cm
from datetime import datetime
from skimage import io, transform

def str2time(instr):
    ymd,hms=instr.split('-')
    return datetime(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]), int(hms[:2]), int(hms[2:4]), int(hms[4:6]))

def str2num(instr):
    return [int(s) for s in instr.split() if s.isdigit()]

def plt_heatmap2d(arr: np.ndarray,cmap:str='viridis'):
    if(cmap not in ['viridis',]):cmap = 'viridis'
    plt.imshow(arr, cmap=cmap)
    plt.colorbar()
    plt.show()

# def plt_trajectory(image_size, ):

def plt_3d_projection(values: np.ndarray, xy_image: np.ndarray=None):
    """
    3D projection of heatmap
    Args:
        values: (h,w,(1)) ndarray of value      
        xy_image: None or (h,w,(ch))
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.figure.subplotpars.left=0.05
    ax.figure.subplotpars.right=0.95
    ax.figure.subplotpars.top=0.05
    ax.figure.subplotpars.bottom=0.95
    h,w=values.shape[:2]
    if(len(values.shape)==3):
        values = values.reshape(values.shape[:2])
    dx = np.arange(w)
    dy = np.arange(h)
    X, Y = np.meshgrid(dx, dy)
    ax.grid(False)
    ax.set_xlim(w, 0)
    ax.set_xticks([])
    ax.set_ylim(0, h)
    ax.set_yticks([])
    ax.set_zlim(0, 1)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    # surf = ax.plot_surface(X, Y, values, cmap=cm.coolwarm,
    #                     linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # # xoy projection
    # cset = ax.contour(X, Y, values, zdir='z', cmap=cm.coolwarm)
    # # xoz projection
    # cset = ax.contour(X, Y, values, zdir='y', cmap=cm.coolwarm)
    # # yoz projection
    # cset = ax.contour(X, Y, values, zdir='x', cmap=cm.coolwarm)
    if(not isinstance(xy_image,type(None))):
        if(len(xy_image.shape)==2):xy_image = np.expand_dims(xy_image,-1)
        # if(xy_image.shape[:2]!=(h,w)):
        xy_image = transform.resize(xy_image,(h,w),preserve_range=False)
        # ax.plot_surface(X, Y, np.array([[-1]]), rstride=2, cstride=2,
        #         facecolors=xy_image)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')

    return fig,ax

def save_image(f_name:str,img:np.ndarray):
    if(os.path.dirname(f_name)==''):
        f_name = os.path.join(os.getcwd(),f_name)
    if(not os.path.exists(os.path.dirname(f_name))):
        os.makedirs(os.path.dirname(f_name))
    if(len(f_name.split('.'))==1):
        f_name += '.jpg'
    mimg.imsave(f_name, img)

def load_single_image(fname,to_torch:bool=False,to_dev=None):
    if(os.path.dirname(f_name)==''):
        f_name = os.path.join(os.getcwd(),f_name)
    if(not os.path.exists(fname)):
        return None
    img = io.imread(fname)
    if(to_torch or not isinstance(to_dev,type(None))):
        img = torch.from_numpy(img)
        if(not isinstance(to_dev,type(None))):
            img = img.to(to_dev)
    return img

if __name__ == "__main__":
    img = io.imread("D:\\development\\workspace\\Dataset\\ICDAR2015\\ch4_test_images\\img_2.jpg")
    
    mask = np.random.uniform(0.0,1.0,(640,640))
    fig,ax = plt_3d_projection(mask,img)
    plt.show(fig)
    pass