import torch
import torch.nn as nn
import itertools as it
from typing import Sequence

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
#POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}

class LinearClassifier(nn.Module):
    """
    Linear Fully Connected Neural Network model
    """
    def __init__(self, in_size, out_classes:int, activation_type: str = "relu", activation_params: dict = None, hidden_dims : list = [196, 49]):
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.activation_type = ACTIVATIONS[activation_type]
        self.activation_params = activation_params

        layers = []
        num_features = self.in_size
        for dim in hidden_dims:
            layers.append(nn.Linear(num_features, dim))
            num_features = dim
            activation = self.activation_type() if self.activation_params is None else self.activation_type(**self.activation_params)
            layers.append(activation)
        layers.append(nn.Linear(num_features, self.out_classes))
        seq = nn.Sequential(*layers)
        self.classifier = seq

    def forward(self, x):
        x = x.reshape(x.size(0), -1) #flatten the input vector (I think flattens to a matrix, might change to only -1 to flatten to a vector)
        print(f'x-shape:{x.shape}')
        size = x.size(0)*x.size(1)
        #x = x.reshape(1,size)
        
        out = self.classifier(x) #FC Classifier
        return out