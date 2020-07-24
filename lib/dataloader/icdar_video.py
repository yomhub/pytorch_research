from lib.dataloader.base import BaseDataset

class ICDARV(BaseDataset):
    def __init__(self, vdo_dir, gt_txt_dir=None, out_box_format='cxywh', normalized=True, transform=None):
        in_box_format = 'yxyx'
        gt_txt_name_lambda = lambda x: "%s_GT.txt"%x

        super(ICDARV,self).__init__(img_dir=vdo_dir, gt_mask_dir=None, gt_txt_dir=gt_txt_dir, in_box_format=in_box_format,
        gt_mask_name_lambda=None, gt_txt_name_lambda=gt_txt_name_lambda, 
        out_box_format=out_box_format, normalized=normalized, transform=transform)

    def read_boxs(self,fname:str):
        txt = open(fname,'r')
        xml = open(fname.split('.')[0]+'.xml','r')
        return None,None