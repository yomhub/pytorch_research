import os
# =================Torch=======================
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# =================Local=======================
from lib.trainer_base import Trainer
from lib.tester_base import Tester
from lib.utils.img_hlp import *
from lib.utils.log_hlp import save_image
from skimage import transform

class CRAFTTrainer(Trainer):
    def __init__(self,  
        train_on_real,
        **params
    ):
        self.train_on_real = bool(train_on_real)
        self.gaussianTransformer = GaussianTransformer()
        self.use_teacher = False

        super(CRAFTTrainer,self).__init__(**params)
    
    def set_teacher(self,mod_dir:str,device='cuda'):
        self.teacher_device = torch.device(device)
        self.teacher = torch.load(mod_dir).float().to(self.teacher_device)
        self.teacher.eval()
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
            for i in range(char_target.shape[0]):
                self._file_writer.add_image('Char Gaussian', char_target[i], step*batch_size+i)               
        else:
            self._file_writer.add_image('Char Gaussian', char_target, step)

        if(len(aff_target.shape)==4):
            for i in range(aff_target.shape[0]):
                self._file_writer.add_image('Affinity Gaussian', aff_target[i], step*batch_size+i)
        else:
            self._file_writer.add_image('Affinity Gaussian', aff_target, step)
        pred = pred[0] if(isinstance(pred,tuple))else pred
        predict_r = torch.unsqueeze(pred[:,0,:,:],1)
        predict_a = torch.unsqueeze(pred[:,1,:,:],1)

        for i in range(predict_r.shape[0]):
            self._file_writer.add_image('Pred char Gaussian', predict_r[i], step*batch_size+i)
            self._file_writer.add_image('Pred affinity Gaussian', predict_a[i], step*batch_size+i)

        return None

    def _step_callback(self,x,y,pred,loss,step,batch_size):
        if(self._file_writer==None):return None
        if(isinstance(loss,list)):
            for i,o in enumerate(loss):
                self._file_writer.add_scalar('Loss/train', o, step*batch_size+i)
        else:
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
        if('video' not in sample):
            x=self._custom_x_input_function(sample,self._device)
            try:
                self._net.init_state()
            except:
                None
            if(self.use_teacher):
                try:
                    img = sample['image'].numpy()
                except:
                    img = sample['image']
                img = np_img_normalize(img)
                with torch.no_grad():
                    gt,_ = self.teacher(torch.from_numpy(img).permute(0,3,1,2).float().to(self.teacher_device))
                gt = torch.from_numpy(gt.to('cpu').numpy()).to(self._device)
                # gt = torch.from_numpy(gt).to(self._device)
                chmap = torch.unsqueeze(gt[:,0,:,:],1)
                afmap = torch.unsqueeze(gt[:,1,:,:],1)
                # chmap -= torch.min(chmap)
                # chmap /= torch.max(chmap)
                # afmap -= torch.min(afmap)
                # afmap /= torch.max(afmap)
                y=chmap,afmap

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
            pred = self._net(x)
            loss = self._loss(pred[0] if(isinstance(pred,tuple))else pred,y)
            loss.backward()
            self._opt.step()
        else:
            self._net.init_state()
            im_size = (sample['height'],sample['width'])
            pointsxy = sample['gt']
            p_keys = list(pointsxy.keys())
            p_keys.sort()
            vdo = sample['video']
            fm_cnt = 0
            loss_list = []
            x_list = []
            y_ch_list = []
            y_af_list = []
            pred_list = []
            while(vdo.isOpened()):
                ret, x = vdo.read()
                if(ret==False):
                    break
                x = transform.resize(x,(640,640),preserve_range=True)
                if(len(x.shape)==3): 
                    x = np.expand_dims(x,0)
                if(len(x_list)<10 and fm_cnt%30==0):
                    x_list.append(torch.from_numpy(x))
                with torch.no_grad():
                    gt,_ = self.teacher(torch.from_numpy(np_img_normalize(x)).float().permute(0,3,1,2).to(self.teacher_device))
                gt = torch.from_numpy(gt.to('cpu').numpy()).to(self._device)
                ch,af = gt[:,0].to('cpu').numpy(),gt[:,1].to('cpu').numpy()
                if(len(y_ch_list)<10 and fm_cnt%30==0):
                    y_ch_list.append(torch.unsqueeze(gt[:,0],1).to('cpu'))
                    y_af_list.append(torch.unsqueeze(gt[:,1],1).to('cpu'))

                y = torch.unsqueeze(gt[:,0],1),torch.unsqueeze(gt[:,1],1)
                word_map = np.where(ch>af,ch,af)

                try:
                    idx = p_keys.index(fm_cnt)
                except:
                    idx = -1

                self._opt.zero_grad()
                pred,_ = self._net(torch.from_numpy(x).float().permute(0,3,1,2).to(self._device))
                if(len(pred_list)<10 and fm_cnt%30==0):
                    pred_list.append(pred.detach().to('cpu'))
                loss = self._loss(pred,y)
                if(idx>0):
                    pred_aff_map = pred[:,2:].to('cpu')
                    dst,src = pointsxy[p_keys[idx]],pointsxy[p_keys[idx-1]]
                    src_box = []
                    dst_box = []
                    for box in dst:
                        ind = np.where(box[0]==src[:,0])[0]
                        if(ind.shape[0]):
                            src_box.append(src[ind[0]][1:].reshape((4,2)))
                            dst_box.append(box[1:].reshape((4,2)))
                    if(src_box):
                        src_box = np_box_resize(np.array(src_box),im_size,ch.shape[-2:],'polyxy')
                        dst_box = np_box_resize(np.array(dst_box),im_size,ch.shape[-2:],'polyxy')

                        aff_list, _ = cv_box_moving_vector(src_box,dst_box)
                        aff_list = aff_list.reshape((-1,6))
                        # (N,1,6,1,1)
                        aff_list = torch.tensor(np.expand_dims(aff_list,(1,3,4)).astype(np.float32),requires_grad=True)
                        loss_box = 0.0
                        box_cnt = 0
                        odet, labels, mapper = cv_getDetCharBoxes_core(word_map[0])
                        labels = torch.from_numpy(labels)
                        for i in range(dst_box.shape[0]):
                            x1= int(np.clip(dst_box[i,:,0].min(),0,ch.shape[-1]-1))
                            x2= max(int(np.clip(dst_box[i,:,0].max(),0,ch.shape[-1])),x1+1)
                            y1= int(np.clip(dst_box[i,:,1].min(),0,ch.shape[-2]-1))
                            y2= max(int(np.clip(dst_box[i,:,1].max(),0,ch.shape[-2])),y1+1)

                            if(torch.max(labels[y1:y2,x1:x2])>0):
                                sub_labels = labels[y1:y2,x1:x2].reshape(-1)
                                sub_map = pred_aff_map[0,:,y1:y2,x1:x2].permute(1,2,0).reshape((-1,6))
                                post = torch.abs(torch.masked_select(sub_map,sub_labels.reshape((-1,1))>0).reshape((-1,6))-aff_list[i].reshape(-1))
                                post = torch.sum(post,dim=-1)
                                loss_box+=torch.mean(post).data
                                box_cnt+=1
                        if(box_cnt>0):
                            loss_box/=box_cnt
                        if(loss_box>0.0):
                            loss+=loss_box
                loss.backward()
                loss_list.append(loss.item())
                self._opt.step()
                # self._net.lstmh = self._net.lstmh.detach()
                # self._net.lstmc = self._net.lstmc.detach()
                self._net.lstmh = self._net.lstmh.grad.data.zero_()
                self._net.lstmc = self._net.lstmc.grad.data.zero_()
                fm_cnt += 1
            vdo.release()
            x=torch.cat(x_list)
            y=torch.cat(y_ch_list),torch.cat(y_af_list)
            loss = loss_list
            pred = torch.cat(pred_list)
        return x,y,pred,loss

class CRAFTTester(Tester):
    def __init__(self,**params):
        self._box_ovlap_th = 0.2
        super(CRAFTTester,self).__init__(**params)

    def _step_callback(self,x,y,pred,loss,cryt,step,batch_size):
        pred = pred[0].to('cpu').detach().numpy()
        scale = (x.shape[2]/pred.shape[2],x.shape[3]/pred.shape[3])
        x = x.permute(0,2,3,1).to('cpu').detach().numpy().astype(np.uint8)
        if(self._file_writer==None):return None
        if(loss!=None): self._file_writer.add_scalar('Loss/test', loss, step)

        wods = np.sum(pred,1)
        for batch,wod in enumerate(wods):
            odet, labels, mapper = cv_getDetCharBoxes_core(wod)

            # odet = np_box_rescale(odet,scale,'polyxy')
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

    def _criterion(self,pred,sample):
        pred = pred[0] if(isinstance(pred,tuple))else pred
        pred = pred.detach().to('cpu').numpy()
        chmap = np.expand_dims(pred[:,0],-1)
        segmap = np.expand_dims(pred[:,1],-1)
        miss_corr = []
        for bh in range(chmap.shape[0]):
            odet, labels, mapper = cv_getDetCharBoxes_core(chmap[bh],segmap[bh])
            gtbox = sample['gt'][bh]
            ovlaps = []
            fp = 0
            for i in range(odet.shape[0]):
                ovlap = cv_box_overlap(gtbox,odet[i])
                ovlap = ovlap.reshape(-1)
                ind = np.argmax(ovlap)
                if(ovlap[ind]>self._box_ovlap_th):
                    ovlaps.append(ovlap[ind])
                elif(ovlap[ind]==-1.0):
                    fp+=1

                odet = np.delete(odet,ind,axis=0)
            tp = len(ovlaps)
            fn = gtbox.shape[0]-tp
            recal = tp/(tp+fn)
            prec = tp/(tp+fp)
            miss_corr.append((recal,prec,2*recal*prec/(recal+prec)))
        
        return np.array(miss_corr)

def train_vdo(net,teacher,vdo_dataset,loss,opt):
    teacher = teacher.float().to('cpu')
    teacher.eval()
    net = net.float().to('cuda')
    loss = loss.float().to('cuda')

    for sample in vdo_dataset:

        im_size = (sample['height'],sample['width'])
        pointsxy = sample['gt']
        p_keys = list(pointsxy.keys()).sort()
        vdo = sample['video']
        cnt = 0
        while(vdo.isOpened()):
            ret, frame = vdo.read()
            if(ret==False):
                break
            if(len(frame)==3): 
                frame = np.expand_dims(frame,0)
            nor_img = np_img_normalize(frame)
            gt,_ = teacher(torch.from_numpy(nor_img).float())
            gt = gt.detach().numpy()
            ch,af = gt[:,0],gt[:,1]
            y = torch.unsqueeze(torch.from_numpy(ch),1).to('cuda'),torch.unsqueeze(torch.from_numpy(af),1).to('cuda')
            word_map = np.where(ch>af,ch,af)

            frame = torch.from_numpy(frame).float()
            try:
                idx = p_keys.index(cnt)
            except:
                idx = -1

            opt.zero_grad()
            pred,_ = net(frame)
            lossv = loss(pred,y)
            pred_aff_map = pred[:,2:].to('cpu')
            if(idx>0):
                dst,src = pointsxy[p_keys[idx]],pointsxy[p_keys[idx-1]]
                src_box = []
                dst_box = []
                for box in dst:
                    ind = np.where(box[0]==src[:,0])[0]
                    if(ind.shape[0]):
                        src_box.append(src[ind[0]][1:].reshape((4,2)))
                        dst_box.append(box[1:].reshape((4,2)))
                if(src_box):
                    src_box = np_box_resize(np.array(src_box),im_size,ch.shape[-2:],'polyxy')
                    dst_box = np_box_resize(np.array(dst_box),im_size,ch.shape[-2:],'polyxy')

                    aff_list, _ = cv_box_moving_vector(src_box,dst_box)
                    # (N,1,6,1,1)
                    aff_list = torch.from_numpy(np.expand_dims(aff_list,(1,3,4)))
                    loss_box = 0.0
                    box_cnt = 0
                    odet, labels, mapper = cv_getDetCharBoxes_core(word_map[0])
                    labels = torch.from_numpy(labels)
                    for i in range(dst_box.shape[0]):
                        x1= int(np.clip(dst_box[i,:,0].min(),0,ch.shape[-1]-1))
                        x2= max(int(np.clip(dst_box[i,:,0].max(),0,ch.shape[-1])),x1+1)
                        y1= int(np.clip(dst_box[i,:,1].min(),0,ch.shape[-2]-1))
                        y2= max(int(np.clip(dst_box[i,:,1].max(),0,ch.shape[-2])),y1+1)

                        if(torch.max(labels[y1:y2,x1:x2])>0):
                            sub_labels = labels[y1:y2,x1:x2].reshape(-1)
                            sub_map = pred_aff_map[0,:,y1:y2,x1:x2].permute(1,2,0).reshape((-1,6))
                            post = torch.abs(torch.masked_select(sub_map,sub_labels>0)-aff_list[i].reshape(6))
                            post = torch.sum(post,dim=-1)
                            loss_box+=torch.mean(post)
                            box_cnt+=1
                    if(box_cnt>0):
                        loss_box/=box_cnt
                    lossv+=loss_box
                    
            lossv.backward()
            opt.step()
            cnt += 1
        vdo.release()



