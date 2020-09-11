import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from matplotlib import cm
from datetime import datetime
from skimage import transform

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
    h,w=values.shape[:2]
    if(len(values.shape)==3):
        values = values.reshape(values.shape[:2])
    dx = np.arange(w)
    dy = np.arange(h)
    X, Y = np.meshgrid(dx, dy)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    surf = ax.plot_surface(X, Y, values, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    cset = ax.contour(X, Y, values, zdir='x', cmap=cm.coolwarm)
    cset = ax.contour(X, Y, values, zdir='y', cmap=cm.coolwarm)
    if(not isinstance(xy_image,type(None))):
        if(len(xy_image.shape)==2):xy_image = np.expand_dims(xy_image,-1)
        if(xy_image.shape[:2]!=(h,w)):
            xy_image = transform.resize(xy_image,(h,w),preserve_range=True)
        ax.imshow(xy_image)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')

    return fig,ax

def save_image(f_name:str,img:np.ndarray):
    if(os.path.dirname(f_name)==''):
        f_name = os.path.join(os.getcwd(),f_name)
    if(len(f_name.split('.'))==1):
        f_name += '.jpg'
    mimg.imsave(f_name, img)