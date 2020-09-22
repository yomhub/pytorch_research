import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from lib.model.vgg16 import VGG16
from lib.utils.net_hlp import init_weights
from lib.model.mobilenet_v2 import MobUNet
from lib.model.lstm import BottleneckLSTMCell

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
        
class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = VGG16(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y, feature


class CRAFT_MOB(nn.Module):
    def __init__(self, width_mult=1.,pretrained=False):
        super(CRAFT_MOB, self).__init__()
        self._mob = MobUNet(width_mult=width_mult)

        self._final_predict = nn.Sequential(
            nn.Conv2d(self._mob.final_predict_ch, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1, stride=1)
        )

        self.init_weights(self._final_predict.modules())

    def init_weights(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_normal_(m.weight.data)
                init.kaiming_normal_(m.weight, mode='fan_out')
                # init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        f = self._mob(x)
        pred = self._final_predict(f)
        return pred,f

class CRAFT_LSTM(nn.Module):
    def __init__(self, width_mult = 1.0, batch_size = 4):
        super(CRAFT_LSTM, self).__init__()
        self._mob = MobUNet(width_mult=width_mult)
        hidden_channels = self._mob.final_predict_ch
        self._lstm = BottleneckLSTMCell(
            input_channels=self._mob.final_predict_ch,
            hidden_channels=self._mob.final_predict_ch,)
        
        self._distribution = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1, stride=1)
        )
        self._aff_map = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 6, kernel_size=1, stride=1)
        )
        self.init_weights(self._distribution.modules())
        self.init_weights(self._aff_map.modules())
        self.lstmh,self.lstmc = self._lstm.init_hidden(batch_size,self._mob.final_predict_ch,(320,320))

    def forward(self,x):
        f = self._mob(x)
        self.lstmh,self.lstmc = self._lstm(f,self.lstmh,self.lstmc)
        score = self._distribution(self.lstmh)
        pred_map = self._aff_map(self.lstmh)

        return torch.cat((score,pred_map),1),self.lstmh

    def init_weights(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_normal_(m.weight.data)
                init.kaiming_normal_(m.weight, mode='fan_out')
                # init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def init_state(self,shape=(320,320),batch_size=1):
        for k,v in self.state_dict().items():
            d = v
            break
        self.lstmh = torch.zeros((batch_size,self._mob.final_predict_ch,shape[0],shape[1]),dtype=d.dtype).to(d.device)
        self.lstmc = torch.zeros((batch_size,self._mob.final_predict_ch,shape[0],shape[1]),dtype=d.dtype).to(d.device)