import torch
import torch.nn as nn
from sklearn import torchviz

# RP layer could be a Linear Layer, with unlearned values that are sampled from N(0,1/n)
# for convenience, we would prefer defining a new layer that calculates gradient only by X and not by W (this might be implemented on a linear layer, check later)

#RP layers will be returned through designated functions, will be returned as matrix tensor

def Gaussian(projected_n:int):
    return torch.normal(mean=torch.zeros(n), std=torch.ones(n)*(1/n))

#TODO: add Li(sparse gaussian) and count sketch random projectors