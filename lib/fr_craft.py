import os
# =================Torch=======================
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
from skimage import transform as TR
from skimage import transform
from matplotlib import path
# =================Local=======================
from lib.trainer_base import Trainer
from lib.tester_base import Tester
from lib.utils.img_hlp import *
from lib.utils.log_hlp import save_image

class CRAFTTrainer(Trainer):
    def __init__(self,  
        train_on_real,
        **params
    ):
        self.train_on_real = bool(train_on_real)
        self.use_teacher = False

        super(CRAFTTrainer,self).__init__(**params)
    
    def set_teacher(self,mod_dir:str,device='cuda'):
        self.teacher_device = torch.device(device)
        self.teacher = torch.load(mod_dir)
        self.teacher = self.teacher.float().to(self.teacher_device)
        print("Load teacher at {}.".format(mod_dir))
        self.teacher.eval()
        self.use_teacher=True

    def _logger(self,sample,x,y,pred,loss,step,batch_size):
        if(self._file_writer==None):return None
        # self._file_writer.add_scalar('Loss/train', loss, step)
        pred = pred[0].to('cpu').detach().numpy() if(isinstance(pred,tuple))else pred.to('cpu').detach().numpy()
        ch_y, af_y = y
        if(len(ch_y.shape)==4):
            if(ch_y.shape[1]==1):
                ch_y = ch_y[:,0,:,:]
                af_y = af_y[:,0,:,:]
            else:
                ch_y = ch_y[:,:,:,0]
                af_y = af_y[:,:,:,0]
        ch_y = ch_y.to('cpu').detach().numpy()
        af_y = af_y.to('cpu').detach().numpy()
        ch_p = pred[:,0,:,:]
        af_p = pred[:,1,:,:]
        if('image' in sample):
            x = sample['image'] if(isinstance(sample['image'],np.ndarray))else sample['image'].numpy()
        else:
            x = x.permute(0,2,3,1).to('cpu').detach().numpy()
        
        batch_size = min(int(pred.shape[0]),2)
        
        for i in range(batch_size):
            af_mask = cv_mask_image(x[i],cv_heatmap(af_y[i]))
            ch_mask = cv_mask_image(x[i],cv_heatmap(ch_y[i]))
            lines = np.ones((af_mask.shape[0],5,3),dtype=af_mask.dtype)*255
            img = np.concatenate((ch_mask,lines,af_mask),axis=-2)
            self._file_writer.add_image('Org:Image', img, step*batch_size+i,dataformats='HWC')
            af_mask = cv_mask_image(x[i],cv_heatmap(af_p[i]))
            ch_mask = cv_mask_image(x[i],cv_heatmap(ch_p[i]))
            lines = np.ones((af_mask.shape[0],5,3),dtype=af_mask.dtype)*255
            img = np.concatenate((ch_mask,lines,af_mask),axis=-2)
            self._file_writer.add_image('Pred:Image', img, step*batch_size+i,dataformats='HWC')

        return None

    def _step_callback(self,sample,x,y,pred,loss,step,batch_size):
        if(self._file_writer==None):return None
        if(isinstance(loss,list)):
            for i,o in enumerate(loss):
                self._file_writer.add_scalar('Loss/train', o, step*batch_size+i)
        else:
            self._file_writer.add_scalar('Loss/train', loss, step)
        for param_group in self._opt.param_groups:
            self._file_writer.add_scalar('LR rate', param_group['lr'], step)
            break
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

    def train_image(self,sample):
        x=self._custom_x_input_function(sample,self._device)
        xnor = torch_img_normalize(x.permute(0,2,3,1)).permute(0,3,1,2)
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
                gt,_ = self.teacher(xnor)

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
                if(sample['box_format']!='polyxy'):
                    cv_boxes = cv_box2cvbox(sample['box'][batch],image_size,sample['box_format'])
                
                ch_mask,af_mask,_,_ = cv_gen_gaussian(cv_boxes,sample['text'][batch],image_size)
                ch_mask_list.append(torch.from_numpy(ch_mask))
                af_mask_list.append(torch.from_numpy(af_mask))
            y = torch.stack(ch_mask_list,0).float().to(self._device), torch.stack(af_mask_list,0).float().to(self._device)
        else:
            y = self._custom_y_input_function(sample,self._device)

        self._opt.zero_grad()
        pred = self._net(xnor)
        
        loss = self._loss(pred[0] if(isinstance(pred,tuple))else pred,y)
        loss.backward()
        self._opt.step()
        return x,y,pred,loss

    def train_video(self,sample):
        try:
            self._net.init_state()
        except:
            None
        flush_stp = 10
        im_size = (sample['height'],sample['width'])
        pointsxy = sample['gt']
        p_keys = list(pointsxy.keys())
        p_keys.sort()
        st_fram = p_keys[0]
        vdo = sample['video']
        fm_cnt = 0
        loss_list = []
        x_list = []
        y_ch_list = []
        y_af_list = []
        pred_list = []
        b_wait_for_flash = False
        while(vdo.isOpened()):
            ret, x = vdo.read()
            if(ret==False):
                break
            if(fm_cnt<st_fram):
                # skip the initial non text frams
                fm_cnt+=1
                continue
            x = transform.resize(x,(640,640),preserve_range=True)
            if(len(x.shape)==3): 
                x = np.expand_dims(x,0)
            if(len(x_list)<10 and fm_cnt%30==0):
                x_list.append(torch.from_numpy(x))
            xnor = torch.from_numpy(np_img_normalize(x)).float().permute(0,3,1,2)
            with torch.no_grad():
                gt,_ = self.teacher(xnor.to(self.teacher_device))
            gt = torch.from_numpy(gt.to('cpu').numpy()).to(self._device)
            if(len(y_ch_list)<10 and fm_cnt%30==0):
                y_ch_list.append(torch.unsqueeze(gt[:,0],1).to('cpu'))
                y_af_list.append(torch.unsqueeze(gt[:,1],1).to('cpu'))

            y = torch.unsqueeze(gt[:,0],1),torch.unsqueeze(gt[:,1],1)

            try:
                idx = p_keys.index(fm_cnt)
            except:
                idx = -1

            self._opt.zero_grad()
            pred,_ = self._net(xnor.to(self._device))
            if(len(pred_list)<10 and fm_cnt%30==0):
                pred_list.append(pred.detach().to('cpu'))
            loss = self._loss(pred,y)
            box_cnt = 0
            if(idx>0):
                pred_aff_map = pred[:,2:].to('cpu')
                ch,af = gt[:,0].to('cpu').numpy(),gt[:,1].to('cpu').numpy()
                word_map = np.where(ch>af,ch,af)
                # dst = t, src = t-1
                dst,src = pointsxy[p_keys[idx]],pointsxy[p_keys[idx-1]]
                src_box = []
                dst_box = []
                loss_box = 0.0
                for box in dst:
                    ind = np.where(box[0]==src[:,0])[0]
                    if(ind.shape[0]):
                        src_box.append(src[ind[0]][1:].reshape((4,2)))
                        dst_box.append(box[1:].reshape((4,2)))
                if(src_box):
                    src_box = np_box_resize(np.array(src_box),im_size,ch.shape[-2:],'polyxy')
                    dst_box = np_box_resize(np.array(dst_box),im_size,ch.shape[-2:],'polyxy')
                    det_box = np.sum(np.abs(src_box-dst_box),axis=(-1,-2))
                    if(det_box.max()>=2.0):
                        aff_list_up, _ = cv_box_moving_vector(dst_box[:,(0,1,3)],src_box[:,(0,1,3)])
                        aff_list_dw, _ = cv_box_moving_vector(dst_box[:,(1,2,3)],src_box[:,(1,2,3)])
                        # aff_list = aff_list.reshape((-1,6))
                        # N,2,3
                        aff_list_up = aff_list_up.astype(np.float32)
                        aff_list_dw = aff_list_dw.astype(np.float32)
                        
                        for i in range(dst_box.shape[0]):
                            if(det_box[i]<=1.0):
                                continue
                            x1= np.clip(dst_box[i,:,0].min(),0,ch.shape[-1]-1)
                            x2= np.clip(dst_box[i,:,0].max(),0,ch.shape[-1])
                            y1= np.clip(dst_box[i,:,1].min(),0,ch.shape[-2]-1)
                            y2= np.clip(dst_box[i,:,1].max(),0,ch.shape[-2])
                            if((int(x2)-int(x1))*(int(y2)-int(y1))<=0):
                                continue
                            tri_up = path.Path(dst_box[i,(0,1,3),:])
                            # tri_dw = path.Path(dst_box[i,(1,2,3),:])
                            dx = np.linspace(x1,x2,int(x2)-int(x1),dtype=np.float32)
                            dy = np.linspace(y1,y2,int(y2)-int(y1),dtype=np.float32)
                            dx,dy = np.meshgrid(dx,dy)
                            # (len_dy,len_dx,3)-x,y,1
                            ds = np.stack([dx,dy,np.ones_like(dy)],-1)
                            # (len_dy,3,len_dx)
                            ds_t = np.moveaxis(ds,-1,-2)
                            # (2,len_dy,len_dx)
                            sr_up = aff_list_up[i].dot(ds_t)
                            sr_dw = aff_list_dw[i].dot(ds_t)
                            # (len_dy,len_dx,2)
                            sr_up = np.moveaxis(sr_up,0,-1)
                            sr_dw = np.moveaxis(sr_dw,0,-1)

                            gt = np.where(
                                tri_up.contains_points(ds[:,:,:2].reshape(-1,2)).reshape(ds.shape[0],ds.shape[1],1),
                                sr_up-ds[:,:,:2],sr_dw-ds[:,:,:2])
                            x1,x2,y1,y2=int(x1),int(x2),int(y1),int(y2)
                            gt = torch.from_numpy(gt)
                            # (2,len_dy,len_dx)->(len_dy,len_dx,2) on cpu
                            sub_map = pred_aff_map[0,:2,y1:y2,x1:x2].permute(1,2,0)
                            sub_map = torch.sum(torch.abs(sub_map-gt),dim=-1)
                            loss_box += torch.mean(sub_map)
                            box_cnt += 1
                    # if(box_cnt>0):
                    #     loss_box/=box_cnt
                try:
                    self._f_train_loger.write("Mask loss:{}, Box loss:{}, Frame: {}.\n".format(loss,loss_box,fm_cnt+1))
                    self._f_train_loger.flush()
                except:
                    None
                if(loss_box>0.0):
                    loss+=loss_box
                loss.backward(retain_graph=True)
                loss_list.append(loss.item())
                self._opt.step()
            try:
                # self._net.lstmh = self._net.lstmh.detach()
                # self._net.lstmc = self._net.lstmc.detach()
                # self._net.lstmh.grad.data.zero_()
                # self._net.lstmc.grad.data.zero_()
                b_wait_for_flash = bool((fm_cnt+1)%flush_stp==0) or b_wait_for_flash
                if(idx==-1 or (b_wait_for_flash and box_cnt==0) or (fm_cnt+1)%(flush_stp*2)==0 or len(src_box)==0):
                    self._net.lstmh.detach_()
                    self._net.lstmc.detach_()
                    b_wait_for_flash=False
                    self._f_train_loger.write("Delete gradient.")
                    self._f_train_loger.flush()
            except:
                None
            fm_cnt += 1
            if(idx==len(p_keys)-1):
                break
        vdo.release()
        try:
            # Freee the memory
            self._net.lstmh.detach_()
            self._net.lstmc.detach_()
        except:
            None
        x=torch.cat(x_list)
        y=torch.cat(y_ch_list),torch.cat(y_af_list)
        loss = loss_list
        pred = torch.cat(pred_list)
        return x,y,pred,loss

    def _train_act(self,sample):
        if('video' not in sample):
            return self.train_image(sample)
        else:
            return self.train_video(sample)

class CRAFTTester(Tester):
    def __init__(self,**params):
        # ICDAR threshold is 0.5, https://rrc.cvc.uab.es/?ch=15&com=tasks
        self._box_ovlap_th = 0.5
        self.score_list = []
        super(CRAFTTester,self).__init__(**params)

    def _step_callback(self,sample,x,y,pred,loss,cryt,step,batch_size):
        if(isinstance(pred,tuple)):
            pred = pred[0]
        self.score_list.append(cryt)
        image = sample['image']
        box_format = sample['box_format'] if(isinstance(sample['box_format'],str))else sample['box_format'][0]
        pred = pred.to('cpu').detach().numpy()
        scale = (pred.shape[2]/x.shape[2],pred.shape[3]/x.shape[3])
        x = x.permute(0,2,3,1).to('cpu').detach().numpy().astype(np.uint8)
        if(loss!=None): self._file_writer.add_scalar('Loss/test', loss, step)
        if(not isinstance(cryt,type(None))):
            recal,prec,f = np.mean(cryt,axis=0)
            self._f_train_loger.write("Recall|Precision|F-mean|step: {}, {}, {}, {}\n".format(recal,prec,f,step))
            self._f_train_loger.flush()
            if(self._file_writer):
                self._file_writer.add_scalar('Recall/test', recal, step)
                self._file_writer.add_scalar('Precision/test', prec, step)
                self._file_writer.add_scalar('F-mean/test', f, step)

        if(not self._isdebug and len(os.listdir(self._logs_path))<12):
            lines = np.ones((pred.shape[2],5,3))*255.0
            lines = lines.astype(x.dtype)
            wods = np.sum(pred,1)
            for batch,wod in enumerate(wods):
                odet, labels, mapper = cv_get_box_from_mask(wod)
                # frame = x[batch]
                frame = image[batch].numpy().astype(np.uint8)
                # frame = np.pad(frame,((0,640-frame.shape[0]),(0,0),(0,0)), 'constant',constant_values=0)
                ch = cv_heatmap(np.expand_dims(pred[batch,0,:,:],-1))
                af = cv_heatmap(np.expand_dims(pred[batch,1,:,:],-1))
                frame = cv2.resize(frame,ch.shape[:2]).astype(np.uint8)

                try:
                    gtbox = sample['box'][batch].numpy()
                except:
                    gtbox = sample['box'][batch]

                if('poly' not in box_format):
                    gtbox = cv_box2cvbox(gtbox,sample['image'].shape[1:-1],box_format)
                gtbox = np_box_rescale(gtbox,scale,'polyxy')

                if(odet.size>0):
                    frame = cv_draw_poly(frame,odet,color=(0,255,0))
                    
                frame = cv_draw_poly(frame,gtbox,color=(255,0,0))
                img = np.concatenate((frame,lines,ch,lines,af),axis=-2)
                save_image(os.path.join(self._logs_path,'{:05d}_im_ch_af.jpg'.format(step*batch_size+batch)),img)

        return None

    def _criterion(self,x,pred,sample):
        """
        Given a batch prediction and sample
        Return:
            (batch,3) with (recall, precision, FP)
        """
        if('box' not in sample):
            return np.array(((-1,-1,-1)))
        pred = pred[0] if(isinstance(pred,tuple))else pred
        pred = pred.detach().to('cpu').numpy()
        chmap = np.expand_dims(pred[:,0],-1)
        segmap = np.expand_dims(pred[:,1],-1)
        box_format = sample['box_format'] if(isinstance(sample['box_format'],str))else sample['box_format'][0]
        miss_corr = []

        for bh in range(chmap.shape[0]):
            odet, labels, mapper = cv_get_box_from_mask(chmap[bh]+segmap[bh])
            if(odet.shape[0]==0):
                miss_corr.append((0.0,0.0,0.0))
                continue
            try:
                gtbox = sample['box'][bh].numpy()
            except:
                gtbox = sample['box'][bh]

            if('poly' not in box_format):
                gtbox = cv_box2cvbox(gtbox,sample['image'].shape[1:-1],box_format)
                gtbox = np_polybox_minrect(gtbox,'polyxy')
            else:
                gtbox = np_polybox_minrect(gtbox,box_format)

            gtbox = np_box_resize(gtbox,x.shape[2:],pred.shape[2:],'polyxy')
            ovlaps = []
            fp = 0
            gtbox_grp = gtbox.copy()
            # (gt,pred)
            # ovlap = cv_box_overlap(gtbox,odet)
            # ovlap = np.where(ovlap>=self._box_ovlap_th,1,0)
            # def max_rote(cur_val,sub_mtx):
            #     max_rote(cur_val,)

            for i in range(odet.shape[0]):
                ovlap = cv_box_overlap(gtbox_grp,odet[i])
                ovlap = ovlap.reshape(-1)
                ind = np.argmax(ovlap)
                if(ovlap[ind]>self._box_ovlap_th):
                    ovlaps.append(ovlap[ind])
                    gtbox_grp = np.delete(gtbox_grp,ind,axis=0)
                    if(gtbox_grp.shape[0]==0):
                        break
                else:
                    fp+=1

            # MLT2017 for non-discare dataset
            prec = len(ovlaps)/odet.shape[0]
            recal = len(ovlaps)/gtbox.shape[0]
            miss_corr.append((recal,prec,2*recal*prec/(recal+prec) if(recal+prec>0)else 0.0))
        
        return np.array(miss_corr)

    def test_on_vdo(self,vdo,gts):
        net = self._net()
        net.eval()
        fm_cnt = 0
        with torch.no_grad():
            while(vdo.isOpened()):
                ret, frame = vdo.read()
                if(ret==False):
                    break
                x = (frame)
                if(len(frame)==3): 
                    frame = np.expand_dims(frame,0)
                nor_img = torch.from_numpy(np_img_normalize(frame)).float().to(self._device)
                pred,_ = net(nor_img)
                pred = pred.to('cpu').numpy()
                ch = cv_heatmap(np.expand_dims(pred[0,0,:,:],-1))
                af = cv_heatmap(np.expand_dims(pred[0,1,:,:],-1))
                frame = frame[0]
                lines = np.ones((frame.shape[0],5,3))*255.0
                lines = lines.astype(frame.dtype)
                img = np.concatenate((frame,lines,ch,lines,af),axis=-2)
                save_image(
                    os.path.join(self._logs_path,'v{:03d}_{:05d}_img_ch_af.jpg'.format(self._current_step,fm_cnt)),
                    img,
                    )
        vdo.release()


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
                    odet, labels, mapper = cv_get_box_from_mask(word_map[0])
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


