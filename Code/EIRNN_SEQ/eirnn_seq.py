import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
# import torch.nn.Parameter as Parameter

_VF = torch._C._VariableFunctions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cpu = torch.device("cpu")
# device = cpu

def rectify(x):
    relu = nn.ReLU()
    return relu(x)
    # return x

class LstmModule(nn.Module):
    def __init__(self, input_units, output_units, hidden_units, batch_size=1, bias = True, 
                 num_chunks = 1, embedding_dim = 50, rectify_inputs=True, dt=0.5, tau=100):
        super(LstmModule, self).__init__()

        self.hidden_units = hidden_units
        self.input_units = input_units
        self.output_units = output_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.rectify_inputs = rectify_inputs
        self.alpha = 0.2

        self.w_in = rectify(nn.Parameter(torch.randn(hidden_units, input_units), requires_grad = True))
        self.w_rec = rectify(nn.Parameter(torch.randn(hidden_units, hidden_units), requires_grad = True))
        self.w_out = rectify(nn.Parameter(torch.randn(output_units, hidden_units), requires_grad = True))
        self.d_rec = nn.Parameter(torch.zeros(hidden_units, hidden_units), requires_grad=False)

        self.relu = nn.ReLU()

        for i in range(hidden_units) :
            if (i < 0.8*hidden_units):
                self.d_rec[i][i] = 1.0
            else:
                self.d_rec[i][i] = -1.0

    def reset_parameters(self):
        """ 
        Initialize parameters (weights) like mentioned in the paper.
        """


    def forward(self, input_, states = None):

        #states : is x_t
        #rectified_states : h_t
        #both of them are hidden units so they have same dimension as of hidden dims

        if states is None:
            states = input_.new_zeros(self.hidden_size, requires_grad=False)
        
        self.w_in = rectify(self.w_in)
        self.w_rec = rectify(self.w_rec)
        self.w_out = rectify(self.w_out)
        alpha = (self.alpha)
        rectified_states = rectify(states)

        # Apply Dale's on recurrent weights
        w_rec_dale = torch.mm(self.w_rec, self.d_rec)

        hidden_update = torch.mm(w_rec_dale, rectified_states)
        input_update = torch.mm(self.w_in, input_)
        states = (1 - alpha) * states + alpha * (hidden_update + input_update) # + noise, to be added
        rectified_states = rectify(states)

        return states, rectified_states

class LSTM(nn.Module):
    def __init__(self, input_units, hidden_units, vocab_size, output_units = 10, embedding_dim = 50, batch_size = 1, dropout=0, num_layers=1):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.dropout = dropout
        self.batch_size = batch_size

        for layer in range(num_layers):
            layer_input_units = input_units if layer == 0 else hidden_units
            cell = LstmModule(input_units = input_units, output_units = output_units, hidden_units = hidden_units, batch_size = batch_size)
            setattr(self, 'cell_{}'.format(layer), cell)
        
        self.embedding_layer = torch.nn.Embedding(vocab_size, input_units)
        self.linear = nn.Linear(hidden_units, 2)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def forward(self, input_, max_time = 50) :
        layer_output = None
        state = None
        max_time = len(input_)
        all_rectified, all_outputs, all_states = [], [], []

        for layer in range(self.num_layers):
            cell = self.get_cell(layer)

            for time in range(max_time):
                # inputx = torch.tensor(input_[time], requires_grad = False)#.to(device)
                states, rectified_states = cell(input_ = self.embedding_layer(input_[time]), hx = state)
                all_rectified.append(rectified_states)              
                all_states.append(states)
                out = self.linear(rectified_states)
                all_outputs.append(out.tolist())
        
        hlast = states
        softmax_out = self.linear(hlast)
        softmax_out = torch.stack([softmax_out], 0)#.to(cpu)
        return softmax_out, all_hidden, all_outputs

