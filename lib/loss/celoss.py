import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlancedCELoss(nn.Module):
    def __init__(self,class_blanced:1):
        super(BlancedCELoss, self).__init__()
        self.mse_loss = nn.CrossEntropyLoss(reduction="none",reduce=False)
        self.class_blanced = class_blanced

    def sample_loss(self,loss,y):
        positive_mask = y > self.positive_th
        num_positive = torch.sum(positive_mask).item()

        k = int(num_positive * self.positive_mult)
        num_all = x.shape[0]
        if k + num_positive > num_all:
            k = int(num_all - num_positive)
        if k < 10:
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

    def forward(self, x, y):
        """
        Args:
            x: prediction (batch,(ch),h,w)
            y: true value (batch,(ch),h,w)
        """
        if(isinstance(x,tuple)):
            x = x[0]
        if(len(x.shape)==3):
            x = x.reshape((x.shape[0],1,x.shape[1],x.shape[2]))
        if(x.shape[-2:]!=y.shape[-2:]):
            if(len(y.shape)==3):
                y = y.reshape((y.shape[0],1,y.shape[1],y.shape[2]))
            y = F.interpolate(y,size=x.shape[2:], mode='nearest')
            y = y[:,0]
        # batch,ch
        x = x.reshape(x.shape[0],x.shape[1],-1)
        # batch,
        y = y.reshape(y.shape[0],-1).astype(torch.long)
        bth_loss = self.mse_loss(x,y)

        loss_every_sample = []
        for i in range(x.shape[0]):
            loss_every_sample.append(self.mse_loss_single(x[i],y[i]))
            
        return torch.stack(loss_every_sample, 0).mean()