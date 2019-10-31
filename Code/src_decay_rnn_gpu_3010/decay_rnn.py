import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
# import torch.nn.Parameter as Parameter

_VF = torch._C._VariableFunctions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#cpu = torch.device("cpu")
#device = cpu

def rectify(x):
    relu = nn.ReLU()
    return relu(x)
    # return x

class LstmModule(nn.Module):


    def __init__(self, input_units, output_units, hidden_units, batch_size=128, bias = True, num_chunks = 1, embedding_dim = 200):
        super(LstmModule, self).__init__()

        input_size = input_units
        hidden_size = hidden_units

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bias = bias
        self.num_chunks = num_chunks
        self.rgate = nn.Parameter(torch.tensor(0.8).cuda())
        self.embedding_dim=embedding_dim
        self.weight_ih = nn.Parameter(torch.Tensor(embedding_dim, num_chunks*hidden_size).cuda())
        self.weight_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size).cuda())
        self.d_rec = nn.Parameter(torch.zeros(num_chunks * hidden_size, hidden_size).cuda(),requires_grad=False)
        
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size).cuda())
            self.bias_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size).cuda())
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
        for name,param in self.named_parameters():
            if name=="rgate":
                param.data  = torch.tensor(0.8).cuda() 

        
        # for k in range(self.batch_size):
        #     for i in range(self.num_chunks*self.hidden_size) :
        #         for j in range (self.hidden_size) :
        #             self.d_rec[k][i][j] = 0.0

   
        for i in range(self.num_chunks) :
            x = i * self.hidden_size
            for j in range(self.hidden_size) :
                if (j < 0.8*self.hidden_size) :
                    self.d_rec[x + j][j] = 1.0
                else :
                    self.d_rec[x + j][j] = -1.0
        

    def forward(self, input_, hx = None):
        """
        An Elman RNN cell with tanh or ReLU non-linearity.
        h' = tanh/relu(w_{ih} x + b_{ih}  +  w_{hh} h + b_{hh})
        """
        # print(self.d_rec)
        # print (self.rgate)

        if hx is None:
            hx = input_.new_zeros(self.num_chunks*self.hidden_size, requires_grad=False)

        #dale_hh = torch.mm(self.relu(self.weight_hh), self.d_rec)

        dale_hh = self.relu(self.weight_hh)*self.d_rec
        
        if (self.bias) :
            print(self.bias_ih.size())
            print(self.weight_ih.size())
            print(input_.size())

            w_x = self.bias_ih + torch.bmm(input_,self.weight_ih)
            w_h = self.bias_hh + torch.matmul(hx,dale_hh)
            
            print(w_x.size())
            print(w_h.size())
        else :
            w_x = torch.bmm(input_,self.weight_ih)
            w_h = torch.bmm(hx,dale_hh)    

        w_w = ((self.rgate) * hx) + ((1-(self.rgate)) * (w_x + w_h))

        h = self.relu(w_w)

        return h

class LSTM(nn.Module):
    def __init__(self, input_units, vocab_size, hidden_units=650,batch_size = 128, embedding_dim = 200, output_units = 10, num_layers = 2, dropout=0.2):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        print(str(embedding_dim)+" "+str(hidden_units)+" "+str(num_layers)+" "+str(batch_size))
        
        for layer in range(num_layers):
            layer_input_units = input_units if layer == 0 else hidden_units
            cell = LstmModule(input_units = self.embedding_dim, output_units = output_units, hidden_units = hidden_units, batch_size = batch_size,embedding_dim=embedding_dim)
            setattr(self, 'cell_{}'.format(layer), cell)
        
        print("input-size "+str(input_units))
        self.embedding_layer = torch.nn.Embedding(vocab_size,self.embedding_dim).cuda()
        #self.embedding_layer = torch.nn.Embedding(vocab_size, self.embedding_dim).cuda()
        #print("vocabsize "+str(vocab_size))
        #self.embedding_layer = torch.nn.Embedding(vocab_size, self.embedding_dim).cuda()


        # self.dropout_layer = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_units, 2)
        # self.softmax = nn.Softmax(dim=0)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def forward(self, input_, max_time = 50) :
        layer_output = None
        all_layers_last_hidden = []
        state = None
        max_time = len(input_)
        all_hidden, all_outputs = [], []
        print("max_time "+str(max_time))
        print("num_layers "+str(self.num_layers))
        print("input_size "+str(input_.size()))

        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            #for time in range(max_time):
            print(input_.size())
            input_emb = self.embedding_layer(input_.long())
            print("input_emb "+str(input_emb.size()))
            #input_=input_.view(input_.shape[0], -1, self.embedding_dim)
            #input_emb = input_emb.view(self.batch_size, self.input_units, 1)

            state = cell(input_ = input_emb, hx = state)
            all_hidden.append(state.tolist())
            out = self.linear(state)
            all_outputs.append(out.tolist())
        
        hlast = state
        softmax_out = self.linear(hlast)
        softmax_out = torch.stack([softmax_out], 0).cuda()
        return softmax_out, all_hidden, all_outputs
