import torch
import numpy as np
from lib.trainer_base import Trainer
from lib.tester_base import Tester
from lib.utils.img_hlp import cv_getDetCharBoxes_core, cv_draw_rect, cv_draw_poly, cv_heatmap, np_box_rescale
from lib.utils.log_hlp import save_image
from torch.utils.tensorboard import SummaryWriter

class CRAFTTrainer(Trainer):
    def __init__(self,  
        **params
    ):
        super(CRAFTTrainer,self).__init__(**params)
    
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
    
    # def calculate()


class CRAFTTester(Tester):
    def __init__(self,  
        train_on_real,
        **params
    ):
        self._train_on_real = train_on_real
        super(CRAFTTester,self).__init__(**params)

    def _step_callback(self,x,y,pred,loss,step,batch_size):
        if(self._file_writer==None):return None
        self._file_writer.add_scalar('Loss/test', loss, step)

        wods = torch.sum(pred,1).to('cpu').detach().numpy()
        for i,wod in enumerate(wods):
            det, labels, mapper = cv_getDetCharBoxes_core(wod)

            odet = np_box_rescale(det,(x.shape[2]/pred.shape[2],x.shape[3]/pred.shape[3]),'polyxy')
            save_image(
                os.path.join(self._logs_path,'{:05d}_im.jpg'.format(step*batch_size+i)),
                cv_draw_poly(x.permute(0,2,3,1).to('cpu').detach().numpy().astype(np.uint8),odet),
                )
            save_image(
                os.path.join(self._logs_path,'{:05d}_ch.jpg'.format(step*batch_size+i)),
                cv_draw_poly(cv_heatmap(np.expand_dims(pred[0,0,:,:].detach().numpy(),-1)),det),
                )
            save_image(
                os.path.join(self._logs_path,'{:05d}_af.jpg'.format(step*batch_size+i)),
                cv_draw_poly(cv_heatmap(np.expand_dims(pred[0,1,:,:].detach().numpy(),-1)),det),
                )

        return None
    
    def _train_act(self,sample):
        x=self._custom_x_input_function(sample,self._device)
        y=self._custom_y_input_function(sample,self._device)
        if(self._train_on_real):
            y=inference_pursedo_bboxes(self._net,self._device,sample['image'],sample['box'],sample['box_format'],sample['text'])
        self._opt.zero_grad()
        pred = self._net(x)
        loss = self._loss(pred,y)
        loss.backward()
        self._opt.step()
        return x,y,pred,loss
        

from lib.utils.img_hlp import cv_crop_image_by_bbox,cv_box2cvbox,cv_watershed
def inference_pursedo_bboxes(net,device,img,boxes,box_format,word):
    boxes = cv_box2cvbox(boxes,img.shape[0:2],box_format)
    for box in boxes:
        word_image, MM = cv_crop_image_by_bbox(img,box,64,64)

        pred,_ = net(torch.from_numpy(np.expand_dims(word_image,0)).permute(0,3,1,2).float().to(device))
        ch_map = pred[0,0,:,:].to('cpu').detach().numpy()
        pursedo_bboxes = cv_watershed(word_image,ch_map)

        pursedo_len = pursedo_len.shape[0]
        real_len = len(word)
        confidence = (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len

        if(confidence<0.5):
            
