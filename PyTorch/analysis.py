import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim
import torch.nn.functional as F
import csv
from pathlib import Path
import pandas as pd
tf.compat.v1.enable_eager_execution()
import numpy as np

cuda0 = torch.device('cuda:0')

class data():
    def __init__(self):
        return
    def load_data(self, x, y):
        
        x_tensor = torch.from_numpy(x)#.to(cuda0)
        y_tensor = torch.from_numpy(1/(1 + np.exp(y)))#.to(cuda0)
        y_tensor = torch.unsqueeze(y_tensor, 1)
        self.train_ds = TensorDataset(x_tensor.float(), y_tensor.float())
        self.train_dl = DataLoader(self.train_ds, batch_size=64)

class model():
    # Layers: [in, l1, l2, ..., out]
    def __init__(self, layers, lr):
        self.lr = lr
        self.layers = []#[nn.ReLU().cuda()]
        for l in range(len(layers)-1):
            in_dim = layers[l]
            next_dim = layers[l+1]
            self.layers.append(nn.Linear(in_dim, next_dim))
            
        self.layers.append(nn.Sigmoid())
            
        
        self.layers = nn.ModuleList(self.layers)#.cuda()
        self.basic = nn.Sequential(*self.layers)#.cuda()
        for m in self.basic.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0, std = .1)
                nn.init.constant_(m.bias, 0)
        for p in self.basic.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -.1, .1))
        self.opt = optim.SGD(self.basic.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def random(self):
        sample = torch.rand([1, 5])#.cuda()
        return self.basic.forward(sample)#.cuda()
    
    def test(self, input):
        sample = torch.tensor(input)#.cuda()
        return self.basic.forward(sample)#.cuda()
    
    def fit(self, trainloader, epochs):
        total_loss = 0
        for e in range(epochs):
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                labels = labels.view(-1)
                
                # zero the parameter gradients
                self.opt.zero_grad()

                # forward + backward + optimize
                outputs = self.basic(inputs)
                #loss = self.criterion(outputs.view(-1), labels.long())
                #loss.backward()
                self.opt.step()

                #total_loss += loss.item()
        return total_loss
print("Imported analysis")