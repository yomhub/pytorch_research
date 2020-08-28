import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRAFTLoss(nn.Module):
    def forward(self,x,y):
        char_target, aff_target = y