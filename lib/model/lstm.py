import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class BottleneckLSTMCell(nn.Module):
	""" Creates a LSTM layer cell
	Arguments:
		input_channels : variable used to contain value of number of channels in input
		hidden_channels : variable used to contain value of number of channels in the hidden state of LSTM cell
	"""
	def __init__(self, input_channels, hidden_channels):
		super(BottleneckLSTMCell, self).__init__()

		assert hidden_channels % 2 == 0

		self.input_channels = int(input_channels)
		self.hidden_channels = int(hidden_channels)
		
		self.W = nn.Conv2d(self.input_channels, self.input_channels, 3, 1, padding=1, groups=self.input_channels)
		self.Wy  = nn.Conv2d(self.input_channels+self.hidden_channels, self.hidden_channels, kernel_size=1)
		self.Wi  = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, groups=self.hidden_channels, bias=False)  
		self.Wbi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbo = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)

		self.Wci = None
		self.Wcf = None
		self.Wco = None
		self._initialize_weights()

	def _initialize_weights(self):
		"""
		Returns:
			initialized weights of the model
		"""
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			
	def forward(self, x, h, c): #implemented as mentioned in paper here the only difference is  Wbi, Wbf, Wbc & Wbo are commuted all together in paper
		"""
		Arguments:
			x : input tensor
			h : hidden state tensor
			c : cell state tensor
		Returns:
			output tensor after LSTM cell 
		"""
		if(h.device!=x.device):
			h = h.to(x.device).type(x.dtype)
			c = c.to(x.device).type(x.dtype)

		y = torch.cat((self.W(x), h),1) #concatenate input and hidden layers
		# --------------------
		i = self.Wy(y) #reduce to hidden layer size
		b = self.Wi(i)	#depth wise 3*3
		ci = torch.sigmoid(self.Wbi(b) + c * self.Wci)
		cf = torch.sigmoid(self.Wbf(b) + c * self.Wcf)
		cc = cf * c + ci * torch.relu(self.Wbc(b))
		co = torch.sigmoid(self.Wbo(b) + cc * self.Wco)
		ch = co * torch.relu(cc)
		# ++++++++++++++++++++
		# b = self.Wi(self.Wy(y))
		# cc = torch.sigmoid(self.Wbf(b) + c * self.Wcf) * c + torch.sigmoid(self.Wbi(b) + c * self.Wci) * torch.relu(self.Wbc(b))
		# ch = torch.sigmoid(self.Wbo(b) + cc * self.Wco) * torch.relu(cc)
		# ==================
		return ch, cc

	def init_hidden(self, batch_size, hidden, shape):
		"""
		Arguments:
			batch_size : an int variable having value of batch size while training
			hidden : an int variable having value of number of channels in hidden state
			shape : an array containing shape of the hidden and cell state 
		Returns:
			cell state and hidden state
		"""
		for k,v in self.state_dict().items():
			d = v
			break
		if self.Wci is None:
			self.Wci = nn.Parameter(torch.rand(1, hidden, shape[0], shape[1],dtype=d.dtype),requires_grad=True)
			self.Wcf = nn.Parameter(torch.rand(1, hidden, shape[0], shape[1],dtype=d.dtype),requires_grad=True)
			self.Wco = nn.Parameter(torch.rand(1, hidden, shape[0], shape[1],dtype=d.dtype),requires_grad=True)
		else:
			assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
			assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'

		return (torch.zeros((batch_size, hidden, shape[0], shape[1]),dtype=d.dtype).to(d.device),
				torch.zeros((batch_size, hidden, shape[0], shape[1]),dtype=d.dtype).to(d.device)
				)

class BottleneckLSTM(nn.Module):
	def __init__(self, input_channels, hidden_channels, height, width, batch_size):
		""" Creates Bottleneck LSTM layer
		Arguments:
			input_channels : variable having value of number of channels of input to this layer
			hidden_channels : variable having value of number of channels of hidden state of this layer
			height : an int variable having value of height of the input
			width : an int variable having value of width of the input
			batch_size : an int variable having value of batch_size of the input
		Returns:
			Output tensor of LSTM layer
		"""
		super(BottleneckLSTM, self).__init__()
		self.input_channels = int(input_channels)
		self.hidden_channels = int(hidden_channels)
		self.cell = BottleneckLSTMCell(self.input_channels, self.hidden_channels)
		(h, c) = self.cell.init_hidden(batch_size, hidden=self.hidden_channels, shape=(height, width))
		self.hidden_state = h
		self.cell_state = c

	def forward(self, input):
		new_h, new_c = self.cell(input, self.hidden_state, self.cell_state)
		self.hidden_state = new_h
		self.cell_state = new_c
		return self.hidden_state
