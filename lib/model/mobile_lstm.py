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
        split_factor = 4,
        base_mbnet_mult = 0.35,
    ):
        super(SLLSTM, self).__init__()
        self._input_shape = (input_shape,input_shape) if(isinstance(input_shape,int))else input_shape
        self._input_shape = (max(1,self._input_shape[0]//32)*32,max(1,self._input_shape[1]//32)*32)
        self._split_factor = int(split_factor)
        out_chs_mult = [int(base_mbnet_mult*320)*self._split_factor/320,float(base_mbnet_mult)]
        
        self._feature_genator_list = [
            MobileNetV2(width_mult=out_chs_mult[0], have_fc=False),
            MobileNetV2(width_mult=out_chs_mult[1], have_fc=False)
        ]
        output_shape = (self._input_shape[0]//32,self._input_shape[1]//32)
        self._split_base_channel = int(320*out_chs_mult[-1])

        self._lstm_chs = self._split_base_channel
        self._lstm = BottleneckLSTMCell(input_channels=self._split_base_channel, hidden_channels=self._lstm_chs)
        (h, c) = self._lstm.init_hidden(batch_size=1, hidden=self._lstm_chs, shape=output_shape)
        self._lstm_h_state_queue = [h]
        self._lstm_h_split_state_queue = [[h]*self._split_factor,[h]]
        self._lstm_c_state_queue = [c]
        self._lstm_c_split_state_queue = [[c]*self._split_factor,[c]]

        num_class = 2
        self._final_predict = nn.Sequential(
            nn.Conv2d(self._lstm_chs *2, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )
        init_weights(self._final_predict.modules())
        self._frame_cont = 0
        self._frame_period = int(frame_period)
        

    def forward(self,x):
        solt = 0 if(self._frame_cont%self._frame_period==0)else 1
        x = self._feature_genator_list[solt](x)

        self._frame_cont += 1
        xs = torch.split(x,self._split_base_channel,1)
        for i in range(len(xs)):
            self._lstm_h_split_state_queue[solt][i],self._lstm_c_split_state_queue[solt][i] = self._lstm(xs[i],
            self._lstm_h_split_state_queue[solt][i],self._lstm_c_split_state_queue[solt][i])

        self._lstm_h_state_queue.append(
            torch.stack(self._lstm_h_split_state_queue[0]+self._lstm_h_split_state_queue[1],-1).sum(dim=-1))
        self._lstm_c_state_queue.append(
            torch.stack(self._lstm_c_split_state_queue[0]+self._lstm_c_split_state_queue[1],-1).sum(dim=-1))

        if(len(self._lstm_h_state_queue)>2):
            self._lstm_h_state_queue.pop(0)
        if(len(self._lstm_c_state_queue)>2):
            self._lstm_c_state_queue.pop(0)

        x = self._final_predict(torch.cat((self._lstm_h_state_queue[-1],self._lstm_c_state_queue[-1]),1))
        
        return x
    