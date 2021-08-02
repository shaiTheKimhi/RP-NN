import torch
import torch.nn as nn
from sklearn import random_projection
import torchviz

#RP matrix: R of shape (d,n), input vector X of shape (B,d) output: XR of shape (B,n)
#R is output of the following functions, given a matrix we will define a Linear projection layer


"""
RP layer could be a Linear Layer, with unlearned values that are sampled from N(0,1/n)
for convenience, we would prefer defining a new layer that calculates gradient only by X and not by W (this might be implemented on a linear layer, check later) 
"""

#RP layers will be returned through designated functions, will be returned as matrix tensor

#TODO: projection should work for batches as well
def Gaussian(original_d:int, projected_n:int):
    n = projected_n
    d = original_d
    transformer = random_projection.GaussianRandomProjection(n)
    return torch.tensor(transformer.fit_transform(torch.eye(d,d)))
    #return torch.normal(mean=torch.zeros(n,d), std=torch.ones(n,d)/(n*d))

#TODO: add Li(sparse gaussian) and count sketch random projectors


#TODO: add an attempt to create PCA dimensional reduction as the 'optimal' solution for comparsion