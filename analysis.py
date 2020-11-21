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

class data():
    def __init__(self, data_path):
        self.path = Path(data_path)
    def read_file(self):
        self.raw_data = pd.read_csv(self.path)
        self.start_date = self.raw_data['Date'][0]
        self.data_length = len(self.raw_data)
        self.columns = len(self.raw_data.columns)
        self.date_range = pd.date_range(self.start_date, periods=self.data_length)
        self.x = self.raw_data.iloc[0:self.data_length, 1:self.columns-2].values
        self.y = self.raw_data["High"].values
        x_tensor = torch.from_numpy(self.x)
        y_tensor = torch.from_numpy(self.y)
        y_tensor = torch.unsqueeze(y_tensor, 1)
        self.train_ds = TensorDataset(x_tensor.float(), y_tensor.float())
        self.train_dl = DataLoader(self.train_ds, batch_size=64)

class model():
    # Layers: [in, l1, l2, ..., out]
    def __init__(self, layers, lr):
        self.lr = lr
        self.layers = [nn.ReLU()]
        for l in range(len(layers)-1):
            in_dim = layers[l]
            next_dim = layers[l+1]
            self.layers.append(nn.Linear(in_dim, next_dim))
        self.layers = nn.ModuleList(self.layers)
        self.basic = nn.Sequential(*self.layers)
        
        self.opt = optim.SGD(self.basic.parameters(), lr=lr)
        self.loss_func = nn.MSELoss(reduction='sum')
        
    def random(self):
        sample = torch.rand([1, 5])
        return self.basic.forward(sample)
    
    def test(self, input):
        sample = torch.tensor(input)
        return self.basic.forward(sample)
    
    def fit(self, data, epochs):
        for e in range(epochs):
            # Forward pass: compute predicted y by passing x to the model. Module objects
            # override the __call__ operator so you can call them like functions. When
            # doing so you pass a Tensor of input data to the Module and it produces
            # a Tensor of output data.
            for xb,yb in data.train_dl:
                y_pred = self.basic(xb)

                # Compute and print loss. We pass Tensors containing the predicted and true
                # values of y, and the loss function returns a Tensor containing the
                # loss.
                loss = self.loss_func(y_pred, yb)
                
                # Zero the gradients before running the backward pass.
                self.basic.zero_grad()

                # Backward pass: compute gradient of the loss with respect to all the learnable
                # parameters of the model. Internally, the parameters of each Module are stored
                # in Tensors with requires_grad=True, so this call will compute gradients for
                # all learnable parameters in the model.
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()


def loss_batch(model, loss_func, xb, yb, opt):
    loss = loss_func(model(xb), torch.max(yb, 1)[1])

    loss.backward()
    opt.step()
    opt.zero_grad()

    return loss.item(), len(xb)

print("Imported analysis")