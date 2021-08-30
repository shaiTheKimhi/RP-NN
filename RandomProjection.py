import torch
import torch.nn as nn
from sklearn import random_projection
import torchviz
from scipy.linalg import hadamard

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
    return torch.normal(mean=torch.zeros(k,d), std=torch.ones(d,k)/k**0.5) # note: 1/k is the variance, 1/k^0.5 is the std


    # transformer = random_projection.GaussianRandomProjection(n)
    # return torch.tensor(transformer.fit_transform(torch.eye(d,d)))

def Achlioptas(original_d:int, projected_k:int):
    k = projected_n
    d = original_d
    probs = torch.tensor([1/6,2/3,1/6])
    achl = torch.distribtutions.Categorical(torch.ones(k,d,3)*probs)
    return (achl.sample()-1) * (3/k)**0.5



# Li(sparse gaussian) random projection
def Li(original_d:int, projected_k:int):
    k = projected_n
    d = original_d
    s = d**0.5 #largest bound for s is d/logd, recommended sqrt(d)
    probs = torch.tensor([1/(2*s),1-1/s,1/(2*s)])
    dist = torch.distribtutions.Categorical(torch.ones(k,d,3)*probs)
    return (dist.sample()-1) * (s/k)**0.5

#SRHT (Li with densifier)
def SRHT(original_d:int, projected_k:int): #for SRHT d must be a power of 2, if not we can padd the number of features to be a power of 2
    k = projected_n
    d = original_d

    B = torch.distribtutions.Bernoulli(torch.ones(k,d)*0.5)
    D = B.sample()*2 - 1
    H = ((1/d)**0.5)*hadamard(d)
    #TODO: return P = DHS



#TODO: add count-sketch random dimensionality reduction using sklearn


#TODO: add an attempt to create PCA dimensional reduction as the 'optimal' solution for comparsion