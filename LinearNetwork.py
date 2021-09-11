import torch
import torch.nn as nn
import itertools as it
from typing import Sequence
import torch.nn.functional as F

import RandomProjection as RP
ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
#POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}
PROJECTIONS = {"Gaussian" : RP.Gaussian, "Achlioptas" : RP.Achlioptas, "Li": RP.Li, "SRHT": RP.SRHT, "CountSketch": RP.CountSketch} ##TODO: add count-sketch

class LinearClassifier(nn.Module):
    """
    Linear Fully Connected Neural Network model
    """
    def __init__(self, in_size, out_classes:int, activation_type: str = "relu", activation_params: dict = None, hidden_dims : list = [196, 49], rp : int = -1, rp_type:str = 'Gaussian', padding=0, dropout=0):
        super().__init__()
        self.padding = padding
        self.in_size = in_size
        self.out_classes = out_classes
        self.activation_type = ACTIVATIONS[activation_type]
        self.activation_params = activation_params
        in_size += padding
        layers = []
        if rp >= 1:
            W = PROJECTIONS[rp_type](in_size, rp)
            W = torch.tensor(W, dtype=torch.float32)
            rplayer = nn.Linear(in_size, rp)
            rplayer.weight = nn.Parameter(W, requires_grad=False)
            layers.append(rplayer)
            self.in_size = rp
        num_features = self.in_size
        for dim in hidden_dims:
            layers.append(nn.Linear(num_features, dim))
            num_features = dim
            activation = self.activation_type() if self.activation_params is None else self.activation_type(**self.activation_params)
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(num_features, self.out_classes))
        layers.append(nn.Softmax(dim=1))
        seq = nn.Sequential(*layers)
        self.classifier = seq

    def forward(self, x): #x of shape (B,1,N,M)
        x = x.reshape(x.size(0), -1) #flatten the input vector (I think flattens to a matrix, might change to only -1 to flatten to a vector)
        x = F.pad(x, (1,self.padding-1))##padding input 
        #x = x.reshape(1,size)
        out = self.classifier(x) #FC Classifier (with RP at start)
        return out