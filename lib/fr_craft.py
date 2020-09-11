import os
# =================Torch=======================
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# =================Local=======================
from lib.trainer_base import Trainer
from lib.tester_base import Tester
from lib.utils.img_hlp import cv_getDetCharBoxes_core, cv_draw_rect, cv_draw_poly, cv_heatmap, np_box_rescale
from lib.utils.log_hlp import save_image
from lib.utils.img_hlp import cv_crop_image_by_bbox,cv_box2cvbox,cv_watershed,GaussianTransformer

class CRAFTTrainer(Trainer):
    def __init__(self,  
        train_on_real,
        **params
    ):
        self.train_on_real = bool(train_on_real)
        self.gaussianTransformer = GaussianTransformer()
        self.use_teacher = False
        super(CRAFTTrainer,self).__init__(**params)
    
    def set_teacher(self,mod_dir:str):
        self.teacher = torch.load(mod_dir)
        self.teacher.eval()
        self.teacher = self.teacher.float().to('cpu')
        self.use_teacher=True

    def _logger(self,x,y,pred,loss,step,batch_size):
        if(self._file_writer==None):return None
        # self._file_writer.add_scalar('Loss/train', loss, step)
        if(len(x.shape)==4):
            self._file_writer.add_image('Image', x[0]/255, step)
        else:
            self._file_writer.add_image('Image', x/255, step)
        char_target, aff_target = y
        if(len(char_target.shape)==4):
            self._file_writer.add_image('Char Gaussian', char_target[0], step)               
        else:
            self._file_writer.add_image('Char Gaussian', char_target, step)

        if(len(aff_target.shape)==4):
            self._file_writer.add_image('Affinity Gaussian', aff_target[0], step)
        else:
            self._file_writer.add_image('Affinity Gaussian', aff_target, step)

        predict_r = torch.unsqueeze(pred[:,0,:,:],0)
        predict_a = torch.unsqueeze(pred[:,1,:,:],0)

        if(len(predict_r.shape)==4):
            self._file_writer.add_image('Pred char Gaussian', predict_r[0], step)
        else:
            self._file_writer.add_image('Pred char Gaussian', predict_r, step)

        if(len(predict_a.shape)==4):
            self._file_writer.add_image('Pred affinity Gaussian', predict_a[0], step)
        else:
            self._file_writer.add_image('Pred affinity Gaussian', predict_a, step)

        return None

    def _step_callback(self,x,y,pred,loss,step,batch_size):
        if(self._file_writer==None):return None
        self._file_writer.add_scalar('Loss/train', loss, step)
        return None
    
    def inference_pursedo_bboxes(self,img:np.ndarray,cv_word_box:np.ndarray,word:str,auto_box_expand:bool=True):
        """
        Inference character level box from word level
        Args:
            img: (h,w,ch) ndarray
            cv_word_box: (4,2) ndarray in cv coordinate
            word: str, word without space (" ")
            auto_box_expand: bool, wither to calculate box by word length if 
                prediction confidence is low. DON'T use this when box is not rectangle
        Return: 
            inference_pursedo_bboxes: character level cv boxes, (N,4,2) with (x,y)
            ch_mask: predicted character mask
            confidence: word confidence
        """
        box = cv_word_box
        word_image, MM = cv_crop_image_by_bbox(img.astype(np.uint8),box,64,64)
        word_image=word_image.astype(np.float32)/255.0
        wd_h,wd_w = word_image.shape[0:2]
        pred,_ = self._net(torch.from_numpy(np.expand_dims(word_image,0)).permute(0,3,1,2).float().to(self._device))
        ch_mask = pred[0,0,:,:].to('cpu').detach().numpy()
        pursedo_bboxes = cv_watershed(word_image,ch_mask)

        pursedo_len = pursedo_bboxes.shape[0]
        real_len = len(word)
        confidence = (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len

        if(confidence<0.45 and auto_box_expand):
            pursedo_bboxes = []
            width_per_char = wd_w/real_len
            for i, char in enumerate(word):
                if(char==' '):
                    continue
                left = i * width_per_char
                right = (i + 1) * width_per_char
                pursedo_bboxes.append([[left, 0], [right, 0], [right, wd_h], [left, wd_h]])
            pursedo_bboxes = np.array(pursedo_bboxes,dtype=np.float)
            pursedo_len = pursedo_bboxes.shape[0]
            confidence = (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len

        for j in range(len(pursedo_bboxes)):
            ones = np.ones((4, 1))
            tmp = np.concatenate([pursedo_bboxes[j], ones], axis=-1)
            I = np.matrix(MM).I
            ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
            pursedo_bboxes[j] = ori[:, :2]

        pursedo_bboxes[:, :, 1] = np.clip(pursedo_bboxes[:, :, 1], 0., img.shape[0] - 1)
        pursedo_bboxes[:, :, 0] = np.clip(pursedo_bboxes[:, :, 0], 0., img.shape[1] - 1)

        return pursedo_bboxes, ch_mask, confidence

    def _train_act(self,sample):
        x=self._custom_x_input_function(sample,self._device)
        if(self.use_teacher):
            gt,_ = self.teacher(sample['image'].permute(0,3,1,2).float())
            gt = gt.to(self._device).detach()
            y = torch.unsqueeze(gt[:,0,:,:],1),torch.unsqueeze(gt[:,1,:,:],1)

        elif(self.train_on_real):
            image_size = sample['image'].shape[1:3]
            ch_mask_list = []
            af_mask_list = []
            for batch in range(sample['image'].shape[0]):
                ch_mask = np.zeros(image_size,dtype=np.float32)
                af_mask = np.zeros(image_size,dtype=np.float32)
                cv_boxes = cv_box2cvbox(sample['box'][batch],image_size,sample['box_format']) if(sample['box_format']!='polyxy')else sample['box'][batch]
                box_list = []
                for i in range(cv_boxes.shape[0]):
                    box,_,_=self.inference_pursedo_bboxes(sample['image'][batch].numpy(),cv_boxes[i],sample['text'][batch][i])
                    box_list.append(box)

                ch_mask = self.gaussianTransformer.generate_region(image_size,box_list,ch_mask)
                af_mask = self.gaussianTransformer.generate_affinity(image_size,box_list,af_mask)
                ch_mask_list.append(torch.from_numpy(np.expand_dims(ch_mask,0)))
                af_mask_list.append(torch.from_numpy(np.expand_dims(af_mask,0)))
            y = torch.stack(ch_mask_list,0).float().to(self._device), torch.stack(af_mask_list,0).float().to(self._device)
        else:
            y = self._custom_y_input_function(sample,self._device)

        self._opt.zero_grad()
        pred,_ = self._net(x)
        loss = self._loss(pred,y)
        loss.backward()
        self._opt.step()
        return x,y,pred,loss

class CRAFTTester(Tester):
    def __init__(self,**params):
        super(CRAFTTester,self).__init__(**params)

    def _step_callback(self,x,y,pred,loss,step,batch_size):
        pred = pred[0].to('cpu').detach().numpy()
        scale = (x.shape[2]/pred.shape[2],x.shape[3]/pred.shape[3])
        x = x.permute(0,2,3,1).to('cpu').detach().numpy().astype(np.uint8)
        if(self._file_writer==None):return None
        if(loss!=None): self._file_writer.add_scalar('Loss/test', loss, step)

        wods = np.sum(pred,1)
        for batch,wod in enumerate(wods):
            det, labels, mapper = cv_getDetCharBoxes_core(wod)

            odet = np_box_rescale(det,scale,'polyxy')
            save_image(
                os.path.join(self._logs_path,'{:05d}_im.jpg'.format(step*batch_size+batch)),
                cv_draw_poly(x[batch],odet),
                )
            save_image(
                os.path.join(self._logs_path,'{:05d}_ch.jpg'.format(step*batch_size+batch)),
                cv_draw_poly(cv_heatmap(np.expand_dims(pred[batch,0,:,:],-1)),det),
                )
            save_image(
                os.path.join(self._logs_path,'{:05d}_af.jpg'.format(step*batch_size+batch)),
                cv_draw_poly(cv_heatmap(np.expand_dims(pred[batch,1,:,:],-1)),det),
                )

        return None

        

