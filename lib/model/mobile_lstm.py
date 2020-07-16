import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.bottleneck_lstm import BottleneckLSTMCell
from lib.model.mobilenet_v2 import MobileNetV2
from lib.utils.net_hlp import init_weights

class SLLSTM(nn.Module):
    def __init__(self,
        input_shape,
        frame_period = 3,
        lstm_state_depth = 256,
        min_depth = 32,
    ):
        super(SLLSTM, self).__init__()
        self._input_shape = input_shape
        self._input_shape = max(1,self._input_shape//64)*64
        self._feature_genator_list = [
            MobileNetV2(input_size=self._input_shape, width_mult=1.4, have_fc=False),
            MobileNetV2(input_size=self._input_shape//2, width_mult=0.35, have_fc=False)
        ]
        output_shape = self._input_shape//32
        output_channel = self._feature_genator_list[0].output_size()
        output_channel_split = output_channel//4

        self._lstm_chs = max(min_depth,lstm_state_depth)
        self._lstm_list = [
            BottleneckLSTMCell(input_channels=output_channel_split, hidden_channels=self._lstm_chs)]
        (h, c) = self._lstm.init_hidden(batch_size=1, hidden=self._lstm_chs, shape=(output_shape, output_shape))
        self._lstm_h_state_queue = [h]
        self._lstm_c_state_queue = [c]

        num_class = 2
        self._final_predict = nn.Sequential(
            nn.Conv2d(self._lstm_chs, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )
        init_weights(self._final_predict.modules())
        self._frame_cont = 0
        self._frame_period = int(frame_period)
        

    def forward(self,x):
        if(self._frame_cont%self._frame_period==0):
            x = self._feature_genator_list[0](x)
        else:
            x = F.interpolate(x,size=int(self._input_shape/2))
            x = self._feature_genator_list[1](x)
        self._frame_cont += 1
        xs = torch.split(x,self._lstm_chs,1)
        for o in xs:
            x,c = self._lstm(o,self._lstm_h_state_queue[-1],self._lstm_c_state_queue[-1])
        self._lstm_h_state_queue.append(x)
        self._lstm_c_state_queue.append(c)

        if(len(self._lstm_h_state_queue)>2):
            self._lstm_h_state_queue.pop(0)
        if(len(self._lstm_c_state_queue)>2):
            self._lstm_c_state_queue.pop(0)

        x = self._final_predict(x)
        
        return x
    