import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math

class Cognitive_CNN(nn.Module):
    def __init__(self, decay_vector, vocab_size, embedding_dim=50, num_classes=1):
        super(Cognitive_CNN, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.decay_vector = torch.tensor(decay_vector, dtype=torch.float)
        self.num_classes=num_classes
        self.vocab_size=vocab_size
        self.embedding_dim = embedding_dim  
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
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
        
    def forward(self, x):
        m = x.shape[0]
        x = self.embedding(x)
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
        return rgate_val
    
    def norm_penalty(self):
        w = self.layer1[0].weight
        norm = torch.norm(torch.norm(torch.norm(w, dim=0), dim=0), dim=0)
        return torch.sum(self.decay_vector*norm)**2
        
    def predict(self, x_test):
        y_pred = self.forward(x_test) >= 0.5
        return y_pred.to(torch.long)

