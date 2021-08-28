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
def Gaussian(original_d:int, projected_k:int):
    k = projected_n
    d = original_d
    
    #returns P:d->k distributes as N(0,1/k)
    return torch.normal(mean=torch.zeros(d,k), std=torch.ones(d,k)/k**0.5) # note: 1/k is the variance, 1/k^0.5 is the std


    # transformer = random_projection.GaussianRandomProjection(n)
    # return torch.tensor(transformer.fit_transform(torch.eye(d,d)))

def Achlioptas(original_d:int, projected_k:int):
    k = projected_n
    d = original_d
    probs = torch.tensor([1/6,2/3,1/6])
    achl = torch.distribtutions.Categorical(torch.ones(d,k,3)*probs)
    return (achl.sample()-torch.ones(d,k)) * (3/k)**0.5



# Li(sparse gaussian) random projection
def Li(original_d:int, projected_k:int):
    k = projected_n
    d = original_d
    s = d**0.5 #largest bound for s is d/logd, recommended sqrt(d)
    probs = torch.tensor([1/(2*s),1-1/s,1/(2*s)])
    dist = torch.distribtutions.Categorical(torch.ones(d,k,3)*probs)
    return (dist.sample()-torch.ones(d,k)) * (s/k)**0.5

#TODO: add sparse gaussian RP



#TODO: add count-sketch random dimensionality reduction using sklearn


#TODO: add an attempt to create PCA dimensional reduction as the 'optimal' solution for comparsion