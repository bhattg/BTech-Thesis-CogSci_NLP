import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math


_VF = torch._C._VariableFunctions

 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 def rectify(x):
    relu = nn.ReLU()
    return relu(x)


class LSTM(nn.Module):
    def __init__(self, input_units, hidden_units, vocab_size, batch_size = 1, embedding_dim = 50, output_units = 10, num_layers = 1, dropout=0):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.embedding_layer = torch.nn.Embedding(vocab_size, input_units)
        self.lstm =  nn.LSTM(input_size =self.embedding_dim, num_layers=self.num_layers, hidden_size= self.hidden_size, dropout=self.dropout, batch_first=True)
        self.linear = nn.LSTM(self.hidden_size, 2)



    def forward(self, input_, max_time = 50) :

    	# input should be of the shape (batch_size, seq_length, embedding_dimension)
    	# assumed input shape =  (batch_size, sequence_lenght)
    	seq_length = input_.shape[1]
    	input_ = self.embedding_layer(input_)
    	input_. = input_.view(self.batch_size, -1, self.embedding_dim)
    	output, hn, cn = self.lstm(input = input_)
    	h_last = output[:, :, -1]
        softmax_out = self.linear(hlast)

        return softmax_out,  hn, cn 
