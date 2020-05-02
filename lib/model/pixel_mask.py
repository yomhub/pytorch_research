import torch
import torch.nn as nn
from torchvision import transforms

class PixelMask(nn.Module):
    def __init__(self):
        super(PixelMask,self).__init__()
            self.resnet = torch.hub.load(
                'pytorch/vision:v0.6.0',
                'resnet50',
                pretrained=True)
            
    def forward(self, x):
        