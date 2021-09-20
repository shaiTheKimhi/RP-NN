import math
import torch
import torch.nn as nn
from sklearn import random_projection
import torchviz
from scipy.linalg import hadamard
import torch.nn.functional as F

#RP matrix: R of shape (d,n), input vector X of shape (B,d) output: XR of shape (B,n)
#R is output of the following functions, given a matrix we will define a Linear projection layer


"""
RP layer could be a Linear Layer, with unlearned values that are sampled from N(0,1/n)
for convenience, we would prefer defining a new layer that calculates gradient only by X and not by W (this might be implemented on a linear layer, check later) 
"""

#RP layers will be returned through designated functions, will be returned as matrix tensor

#TODO: projection should work for batches as well
def Gaussian(original_d:int, projected_k:int):
    k = projected_k
    d = original_d
    
    #returns P:d->k distributes as N(0,1/k)
    return torch.normal(mean=torch.zeros(k,d), std=torch.ones(k,d)/k**0.5) # note: 1/k is the variance, 1/k^0.5 is the std


    # transformer = random_projection.GaussianRandomProjection(n)
    # return torch.tensor(transformer.fit_transform(torch.eye(d,d)))

def Achlioptas(original_d:int, projected_k:int):
    k = projected_k
    d = original_d
    probs = torch.tensor([1/6,2/3,1/6])
    achl = torch.distributions.Categorical(torch.ones(k,d,3)*probs)
    return (achl.sample()-1) * (3/k)**0.5



# Li(sparse gaussian) random projection
def Li(original_d:int, projected_k:int):
    k = projected_k
    d = original_d
    s = d**0.5 #largest bound for s is d/logd, recommended sqrt(d)
    probs = torch.tensor([1/(2*s),1-1/s,1/(2*s)])
    dist = torch.distributions.Categorical(torch.ones(k,d,3)*probs)
    return (dist.sample()-1) * (s/k)**0.5

#SRHT (Li with densifier) 
def SRHT(original_d:int, projected_k:int, dataset_n:int = 12e6): #for SRHT d must be a power of 2, if not we can padd the number of features to be a power of 2
    #TODO: add padding to vector for k to be power of 2
    k = projected_k
    d = original_d
    n = dataset_n
    q = (math.log(n, 2)**2)/d # q is O(log^2(n)/d) we chose 10 as base of log for convenience
    q = 1 if q > 1 else q if q >= 0 else 0 # q is probability value

    B = torch.distributions.Bernoulli(torch.ones(d)*0.5)
    D = torch.diag(B.sample()*2 - 1) # D is d X d
    H = torch.tensor(((1/d)**0.5)*hadamard(d),dtype=torch.float32) # H is d X d
    S = torch.normal(mean=torch.zeros(d,k), std=torch.ones(d,k)/q**0.5) * torch.distributions.Bernoulli(torch.ones(d,k)*0.5).sample() # S is d X k
    return S.T@H@D/k**0.5



#count-sketch 
def CountSketch(original_d: int, projected_k:int):
    k = projected_k
    d = original_d
    B = torch.distributions.Bernoulli(torch.ones(d)*0.5)
    D = torch.diag(B.sample()*2 - 1) # D is d X d
    probs = torch.ones(k)/k
    m = torch.distributions.Categorical(torch.ones(d,k)*torch.tensor(probs))
    A = F.one_hot(m.sample()+1, num_classes=k+1)
    C = A.T[torch.arange(A.size(1))!=0] # C is d X k
    C = torch.tensor(C, dtype=torch.float32)
    D = torch.tensor(D, dtype=torch.float32)
    return (C @ D) #possible: calc C.T @ D could be easier to calculate



#TODO: add an attempt to create PCA dimensional reduction as the 'optimal' solution for comparsion
def get_dataset_matrix(dataset):
    l = []
    for item in dataset:
        item = item[0].reshape(item[0].size(0), -1).tolist() #flatten the input and then to list
        l.append(item)
    M = torch.tensor(l)
    return M.reshape(M.shape[0], -1)
def get_pca(dataset_matrix):
    return 0
    