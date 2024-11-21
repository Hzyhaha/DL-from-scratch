import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor,Compose
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import sys
sys.path.append('..')
# load MINIST
def toarray(data):
    loader = DataLoader(data)
    sampler = [i for i in range(300)]
    loader = DataLoader(data,sampler=sampler)
    
    X,Y = [],[]
    for x,y in loader:
        ax = x.reshape(-1).detach().numpy()
        ay = y.reshape(-1).detach().numpy()
        X.append(ax)
        Y.append(ay)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def load_minist(flag='train'):
    if flag == 'train':
        train_data = datasets.MNIST(
            root='../data',
            transform=ToTensor(),
            download=True,
            train = True,
            target_transform=Compose([
                torch.tensor,
                lambda x: one_hot(x,10)
            ])
        )
        x,x_label = toarray(train_data)
        return x,x_label
    

    if flag == 'test':
        test_data = datasets.MNIST(
            root='../data',
            transform=ToTensor(),
            download=True,
            train = False,
            target_transform=Compose([
                torch.tensor,
                lambda x: one_hot(x,10)
            ])
        )
        t,t_label = toarray(test_data)
        return t,t_label
    
x,x_label = load_minist()
t,t_label = load_minist('test')
print(x.shape)

from common.multi_layer_net import MultiLayerNet

net = MultiLayerNet(input_size=784,
                    hidden_size_list=[100 for k in range(6)],
                    output_size=10)

from optimizer import SGD
optimizer = SGD()


batch_size = 100
train_acc = []
test_acc = []
from tqdm import tqdm
iter_per_epoch = x.shape[0]/batch_size
max_epoch = 200
epoch_cnt = 0
for i in tqdm(range(1000000)):
    mask = np.random.choice(x.shape[0],batch_size)
    train_data = x[mask]
    train_label = x_label[mask]

    grads = net.gradient(train_data,train_label)
    optimizer.update(net.params,grads)

    if i % iter_per_epoch == 0:
        train_acc.append(net.accuracy(x,x_label))
        test_acc.append(net.accuracy(t,t_label))
        epoch_cnt = epoch_cnt + 1

    if epoch_cnt == max_epoch: break



import pylab as plt
plt.plot(train_acc)
plt.plot(test_acc)
plt.show()