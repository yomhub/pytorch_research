import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mimg
from matplotlib import cm
from datetime import datetime
from skimage import io, transform
from collections import OrderedDict
import cv2

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

def plt_show_images(plt_w:int,plt_h:int,image_list:list,title_list:list=None,hide_axis:bool=True):
    """
    PLT image show helpper
    Return fig and ax tuple
    """
    fig,ax_tup = plt.subplots(plt_w,plt_h,sharey=True)
    ax_tup = ax_tup.reshape(-1)
    if(hide_axis):
        for i in range(len(ax_tup)):
            ax_tup[i].get_yaxis().set_visible(False)
            ax_tup[i].get_xaxis().set_visible(False)
    for i,img in enumerate(image_list):
        if(img.dtype==np.float):
            ax_tup[i].imshow((img*255).astype(np.uint8),'gray')
        else:
            ax_tup[i].imshow(img)
        if(title_list and i<len(title_list)):
            ax_tup[i].set_title(title_list[i])

    return fig,ax_tup

def save_image(f_name:str,img:np.ndarray,cmap='rbg'):
    if(os.path.dirname(f_name)==''):
        f_name = os.path.join(os.getcwd(),f_name)
    if(not os.path.exists(os.path.dirname(f_name))):
        os.makedirs(os.path.dirname(f_name))
    if(len(f_name.split('.'))==1):
        f_name += '.jpg'
    # mimg.imsave(f_name, img)
    cv2.imwrite(f_name, img)

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

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def load_net_from_pth(net,pth_dir:str):
    print("Load network from pth at {}".format(pth_dir))
    if(not os.path.exists(pth_dir)):
        print("Error: pth file not exist.")
        return net
    para = copyStateDict(torch.load(pth_dir))
    for o in para:
        net.state_dict()[o]=para[o]
    return net
    
def log_net_hock(net):
    for module in net.modules():
        module.register_forward_hook(lambda m, input, output: 
            print("{}:{}->{}".format(m.__class__.__name__,
                input[0].shape if(isinstance(input,tuple))else input.shape,
                output[0].shape if(isinstance(input,tuple))else output.shape,
                )))

def concatenate_images(images:list,direction:str='h',line_wide:int=3):
    h,w=images[0].shape[0],images[0].shape[1]
    for i in range(len(images)):
        if(len(images[i].shape)==2):
            if(images[i].dtype in [np.uint8,np.uint16,np.int8,np.int16]):
                images[i] = images[i].astype(np.uint8)
            else:
                images[i] = (images[i]*255).astype(np.uint8)
            images[i] = np.expand_dims(images[i],-1)
            images[i] = np.concatenate((images[i],images[i],images[i]),-1)
        h,w = min(h,images[i].shape[0]),min(w,images[i].shape[1])
    if('h' in direction.lower()):
        line = np.ones((h,line_wide,3),dtype=np.uint8)*255
        ax = 1
    else:
        line = np.ones((line_wide,w,3),dtype=np.uint8)*255
        ax = 0
    rets = []
    for i in range(len(images)):
        if(images[i].shape[0]!=h or images[i].shape[1]!=w):
            images[i] = cv2.resize(images[i],(w,h))
        rets.append(images[i])
        rets.append(line)
    rets.pop(-1)

    return np.concatenate(rets,ax)


if __name__ == "__main__":
    img = io.imread("D:\\development\\workspace\\Dataset\\ICDAR2015\\ch4_test_images\\img_2.jpg")
    
    mask = np.random.uniform(0.0,1.0,(640,640))
    fig,ax = plt_3d_projection(mask,img)
    plt.show(fig)
    pass