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
    def __init__(self,positive_mult = 3):
        super(MSE_OHEM_Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="none")
        self._positive_mult = float(positive_mult)
    
    def mse_loss_single(self,img,target):
        img = img.view(1, -1)
        target = target.view(1, -1)
        positive_mask = target > 0
        sample_loss = self.mse_loss(img, target)

        num_positive = int(positive_mask.sum().data.cpu().item())

        k = int(num_positive * self._positive_mult)
        num_all = img.shape[1]
        if k + num_positive > num_all:
            k = int(num_all - num_positive)
        if k < 10:
            avg_sample_loss = sample_loss.mean()
        else:
            positive_loss = torch.masked_select(sample_loss, positive_mask)
            negative_loss = torch.masked_select(sample_loss, target <= 0.0)
            negative_loss_topk, _ = torch.topk(negative_loss, k)
            avg_sample_loss = positive_loss.mean() + negative_loss_topk.mean()

        return avg_sample_loss

    def forward(self, x, y):
        char_target, aff_target = y
        x = x[0]
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