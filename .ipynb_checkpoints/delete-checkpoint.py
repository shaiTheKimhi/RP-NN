from RandomProjection import SRHT
import torch

x = torch.randn(1,1024)
A = SRHT(1024,100)
y = A@x.T
print(y.norm(), x.norm())