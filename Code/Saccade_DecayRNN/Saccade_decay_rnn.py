import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
# import torch.nn.Parameter as Parameter

_VF = torch._C._VariableFunctions

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
device = cpu

def rectify(x):
    relu = nn.ReLU()
    return relu(x)
    # return x

class LstmModule(nn.Module):


    def __init__(self, input_units, output_units, hidden_units, batch_size, bias = True, num_chunks = 1, embedding_dim = 50):
        super(LstmModule, self).__init__()

        input_size = input_units
        hidden_size = hidden_units
        self.rgate = None
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bias = bias
        self.num_chunks = num_chunks
        self.weight_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        self.d_rec = nn.Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size), requires_grad=False)

        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

        for i in range(self.num_chunks*self.hidden_size) :
            for j in range (self.hidden_size) :
                self.d_rec[i][j] = 0.0

        for i in range(self.num_chunks) :
            x = i * self.hidden_size
            for j in range(self.hidden_size) :
                if (j < 0.8*self.hidden_size) :
                    self.d_rec[x + j][j] = 1.0
                else :
                    self.d_rec[x + j][j] = -1.0



    def forward(self, input_,rgate_val ,hx = None):
        """
        An Elman RNN cell with tanh or ReLU non-linearity.
        h' = tanh/relu(w_{ih} x + b_{ih}  +  w_{hh} h + b_{hh})
        """
        # print(self.d_rec)
        # print (self.rgate)
        self.rgate = rgate_val

        if hx is None:
            hx = input_.new_zeros(self.hidden_size, requires_grad=False)

        dale_hh = torch.mm(self.relu(self.weight_hh), self.d_rec)
        if (self.bias) :
            w_x = torch.addmv(self.bias_ih, self.weight_ih, input_)
            w_h = torch.addmv(self.bias_hh, dale_hh, hx)
        else :
            w_x = torch.mv(self.weight_ih, input_)
            w_h = torch.mv(dale_hh, hx)
            
        w_w = ((self.rgate) * hx) + ((1-(self.rgate)) * (w_x + w_h))

        h = self.relu(w_w)

        return h

class LSTM(nn.Module):
    def __init__(self, input_units, hidden_units, vocab_size, batch_size = 1, embedding_dim = 50, output_units = 10, num_layers = 1, dropout=0, num_classes=1):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size


        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.num_classes=num_classes
        self.vocab_size=vocab_size
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size= (50,9)),
            self.relu
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(1, 11)),
            self.relu
        )
        self.linear1 = nn.Linear(52, 35)
        self.linear2 = nn.Linear(35, 15)
        self.linear3 = nn.Linear(15, 1)


        for layer in range(num_layers):
            layer_input_units = input_units if layer == 0 else hidden_units
            cell = LstmModule(input_units = input_units, output_units = output_units, hidden_units = hidden_units, batch_size = batch_size)
            setattr(self, 'cell_{}'.format(layer), cell)
        
        self.embedding_layer = torch.nn.Embedding(vocab_size, input_units)
        # self.dropout_layer = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_units * num_layers, 2)
        # self.softmax = nn.Softmax(dim=0)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    # def get_rgate_val(self):
    #     rgate_list=[]  # A list that contains the rgate values layer wise 
    #     for layer in range(self.num_layers):
    #         cell = self.get_cell(layer)
    #         rgate_list.append(cell.rgate)
    #     return rgate_list


    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def forward(self, input_, max_time = 50) :

        x = input_
        m = 1
        x = self.embedding_layer(x)
        x = x.view(m, 1, self.embedding_dim, -1)
        x = nn.ZeroPad2d((10, 10, 0, 0))(x) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(m,-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        rgate_val = x.view([])



        layer_output = None
        all_layers_last_hidden = []
        state = None
        max_time = len(input_)
        all_hidden, all_outputs = [], []

        for layer in range(self.num_layers):
            cell = self.get_cell(layer)

            for time in range(max_time):
                # print(time)
                # inputx = torch.tensor(input_[time], requires_grad = False)#.to(device)
                state = cell(input_ = self.embedding_layer(input_[time]), rgate_val=rgate_val ,hx = state)
                all_hidden.append(state.tolist())
                out = self.linear(state)
                all_outputs.append(out.tolist())
        
        hlast = state
        softmax_out = self.linear(hlast)
        softmax_out = torch.stack([softmax_out], 0)#.to(cpu)
        return softmax_out, all_hidden, all_outputs


