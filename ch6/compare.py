import numpy as np
import sys
sys.path.append('../common')
from twolayernn import TwoLayerNet
from optimizer import *
from load_data import load_minist
from tqdm import tqdm
import pylab as plt


x_train,t_train,x_test,t_test = load_minist()

def train(class_optimizer):
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)    
    iter_num = 5000
    trian_size = x_train.shape[0]
    batch_size = 100
    lr = 0.1
    loss_list = []
    optimizer = class_optimizer(lr)
    for i in tqdm(range(iter_num)):
        idx_set = np.random.choice(trian_size,batch_size)
        data = x_train[idx_set]
        label = t_train[idx_set]

        grads = net.gradient(data,label)
        pararms = net.params
            
        optimizer.update(pararms,grads)
            
        if i % 100 == 0:
            loss_list.append(net.loss(x_train,t_train))

    
    plt.plot(loss_list)
    print('accuracy: ',net.accuracy(x_test,t_test))

train(SGD)
train(Momentum)
train(AdaGrad)

plt.legend(['SGD','Momentum','AdaGrad'])
plt.show()
