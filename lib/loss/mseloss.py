import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Maploss(nn.Module):
    def __init__(self, use_gpu = True):

        super(Maploss,self).__init__()

    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1))*0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        internel = batch_size
        for i in range(batch_size):
            average_number = 0
            loss = torch.mean(pre_loss.view(-1)) * 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= 0.1)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < 0.1)])
                    average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < 0.1)], 3*positive_pixel)[0])
                    average_number += 3*positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                average_number += 500
                sum_loss += nega_loss
            #sum_loss += loss/average_number

        return sum_loss

    def forward(self, gh_label, gah_label, p_gh, p_gah, mask):
        gh_label = gh_label
        gah_label = gah_label
        p_gh = p_gh
        p_gah = p_gah
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert p_gh.size() == gh_label.size() and p_gah.size() == gah_label.size()
        loss1 = loss_fn(p_gh, gh_label)
        loss2 = loss_fn(p_gah, gah_label)
        loss_g = torch.mul(loss1, mask)
        loss_a = torch.mul(loss2, mask)

        char_loss = self.single_image_loss(loss_g, gh_label)
        affi_loss = self.single_image_loss(loss_a, gah_label)
        return char_loss/loss_g.shape[0] + affi_loss/loss_a.shape[0]

class MSE_OHEM_Loss(nn.Module):
    def __init__(self,positive_mult = 3,positive_th:float = 0.0):
        super(MSE_OHEM_Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="none", size_average=False)
        self.positive_mult = float(positive_mult)
        self.positive_th = float(positive_th)
    
    def mse_loss_single(self,img,target):
        img = img.view(1, -1)
        target = target.view(1, -1)
        positive_mask = target > self.positive_th
        sample_loss = self.mse_loss(img, target)

        num_positive = int(positive_mask.sum().data.cpu().item())

        k = int(num_positive * self.positive_mult)
        num_all = img.shape[1]
        if k + num_positive > num_all:
            k = int(num_all - num_positive)
        if k < 10:
            avg_sample_loss = sample_loss.mean()
        else:
            positive_loss = torch.masked_select(sample_loss, positive_mask)
            negative_loss = torch.masked_select(sample_loss, target <= self.positive_th)
            negative_loss_topk, _ = torch.topk(negative_loss, k)
            avg_sample_loss = positive_loss.mean() + negative_loss_topk.mean()

        return avg_sample_loss

    def forward(self, x, y):
        """
        Args:
            x: pred (batch,2,h,w),feature
            y: char_target ,aff_target (batch,1,h,w)
        """
        char_target, aff_target = y
        if(isinstance(x,tuple)):
            x = x[0]
        if(len(x.shape)==3):
            x=torch.reshape(x,(1,x.shape[0],x.shape[1],x.shape[2]))
        loss_every_sample = []
        batch_size = x.shape[0]
        # x = x.permute(0,2,3,1)
        predict_r = x[:,0,:,:]
        predict_a = x[:,1,:,:]
        char_target = F.interpolate(char_target,size=predict_r.shape[1:], mode='bilinear', align_corners=False)
        aff_target = F.interpolate(aff_target,size=predict_a.shape[1:], mode='bilinear', align_corners=False)

        for i in range(batch_size):
            char_loss = self.mse_loss_single(predict_r[i],char_target[i])
            aff_loss = self.mse_loss_single(predict_a[i],aff_target[i])
            loss_every_sample.append(char_loss+aff_loss)
            
        return torch.stack(loss_every_sample, 0).mean()
        

class MSE_2d_Loss(nn.Module):
    def __init__(self,positive_mult = 3,positive_th:float = 0.0,pixel_sum:bool=False):
        super(MSE_2d_Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="none", size_average=False,reduce=False)
        self.positive_mult = float(positive_mult)
        self.pixel_sum = bool(pixel_sum)
        self.positive_th = positive_th

    def mse_loss_single(self,x,y,weight_mask=None):
        positive_mask = y > self.positive_th
        sample_loss = self.mse_loss(x, y)
        if(not isinstance(weight_mask,type(None))):
            sample_loss*=weight_mask
        num_positive = torch.sum(positive_mask).item()

        k = int(num_positive * self.positive_mult)
        num_all = x.shape[0]
        if(k + num_positive >= num_all or k<=10):
            sample_loss = torch.sum(sample_loss) if(self.pixel_sum)else torch.mean(sample_loss)
        else:
            positive_loss = torch.masked_select(sample_loss, positive_mask)
            negative_loss = torch.masked_select(sample_loss, y <= self.positive_th)
            negative_loss_topk, _ = torch.topk(negative_loss, k)
            if(self.pixel_sum):
                sample_loss = torch.sum(positive_loss) + torch.sum(negative_loss_topk)
            else:
                sample_loss = torch.mean(positive_loss) + torch.mean(negative_loss_topk)

        return sample_loss

    def forward(self, *args):
        """
        Args:
            x: prediction (batch,(ch),h,w)
            y: true value (batch,(ch),h,w)
            weight_mask (optical): loss weight mask in [0,1]
        """
        x,y = args[0],args[1]

        if(isinstance(x,tuple)):
            x = x[0]
        if(len(x.shape)==3):
            x = x.reshape((x.shape[0],1,x.shape[1],x.shape[2]))
        if(len(y.shape)==3):
            y = y.reshape((y.shape[0],1,y.shape[1],y.shape[2]))

        y = F.interpolate(y,size=x.shape[2:], mode='bilinear', align_corners=False)

        b_have_weight_mask = False
        if(len(args)>2):
            b_have_weight_mask = True
            weight_mask = args[2]
            if(len(weight_mask.shape)==3):
                weight_mask = weight_mask.reshape((weight_mask.shape[0],1,weight_mask.shape[1],weight_mask.shape[2]))
            weight_mask = F.interpolate(weight_mask,size=x.shape[2:], mode='bilinear', align_corners=False)
            weight_mask = weight_mask.reshape(weight_mask.shape[0],-1)

        x = x.reshape(x.shape[0],-1)
        y = y.reshape(y.shape[0],-1)
        loss_every_sample = []
        for i in range(x.shape[0]):
            loss_every_sample.append(self.mse_loss_single(x[i],y[i],weight_mask[i] if(b_have_weight_mask)else None))
            
        return torch.stack(loss_every_sample, 0).mean()

class MASK_MSE_LOSS(nn.Module):
    def __init__(self,positive_mult = 3,positive_th:float = 0.0, 
        positive_id:int=2,negative_id:int = 1,ignore_id:int=0):
        super(MASK_MSE_LOSS, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="none", size_average=False,reduce=False)

        self.positive_mult = float(positive_mult)
        self.positive_th = positive_th

        self.positive_id = int(positive_id)
        self.negative_id = int(negative_id)
        self.ignore_id = int(ignore_id)
    
    def mse_loss_single(self,x,y,idmask):
        positive_mask = idmask == self.positive_id
        num_positive = torch.sum(positive_mask).item()
        negative_mask = idmask == self.negative_id
        num_negative = torch.sum(negative_mask).item()

        sample_loss = self.mse_loss(x, y)

        k = int(num_positive * self.positive_mult)
        num_all = num_positive+num_negative

        if(k + num_positive >= num_all or k<=10):
            sample_loss = torch.mean(torch.masked_select(sample_loss,idmask != self.ignore_id))
        else:
            positive_loss = torch.masked_select(sample_loss, positive_mask)
            negative_loss = torch.masked_select(sample_loss, negative_mask)
            negative_loss_topk, _ = torch.topk(negative_loss, k)
            sample_loss = torch.mean(positive_loss) + torch.mean(negative_loss_topk)

        return sample_loss

    def forward(self, x, y, idmask):
        if(len(x.shape)==3):
            x = x.reshape((x.shape[0],1,x.shape[1],x.shape[2]))
        if(len(y.shape)==3):
            y = y.reshape((y.shape[0],1,y.shape[1],y.shape[2]))
        if(len(idmask.shape)==3):
            idmask = idmask.reshape((idmask.shape[0],1,idmask.shape[1],idmask.shape[2]))

        if(y.shape[2:]!=x.shape[2:]):
            y = F.interpolate(y,size=x.shape[2:], mode='bilinear', align_corners=False)
        if(idmask.shape[2:]!=x.shape[2:]):
            idmask = F.interpolate(idmask.float(),size=x.shape[2:], mode='nearest').type(torch.int)
        
        x = x.reshape(x.shape[0],-1)
        y = y.reshape(y.shape[0],-1)
        idmask = idmask.reshape(idmask.shape[0],-1)

        loss_every_sample = []
        for i in range(x.shape[0]):
            loss_every_sample.append(self.mse_loss_single(x[i],y[i],idmask[i]))

        return torch.stack(loss_every_sample, 0).mean()