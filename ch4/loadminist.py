import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
# load MINIST
data = datasets.MNIST(
    root='data',
    transform=ToTensor(),
    download=False
)

sampler = [i for i in range(100)]
loader = DataLoader(data,sampler=sampler)
X,Y = [],[]
for x,y in loader:
   ax = x.reshape(-1).detach().numpy()
   ay = y.detach().numpy()
   X.append(ax)
   Y.append(ay)
X = np.array(X)
Y = np.array(Y)
