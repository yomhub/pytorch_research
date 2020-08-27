import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from datetime import datetime

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

def save_image(img:np.ndarray,f_name:str='image.jpg'):
    if(os.path.dirname(f_name)==''):
        f_name = os.path.join(os.getcwd(),f_name)
    mimg.imsave(f_name, img)