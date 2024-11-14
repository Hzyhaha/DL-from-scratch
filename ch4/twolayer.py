import numpy as np

import sys,os
sys.path.append('.')
from common.activation_func import *
from common.loss_func import *
from common.diffraction import numerical_diff
from common.load_data import load_minist
from tqdm import tqdm

class TwoLayerNet :
    def __init__(self,
                 input_size,hidden_size,output_size,
                 init_std = 0.01) -> None:
        self.params = {}
        self.params['w1'] = np.random.randn(input_size,hidden_size) * init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = np.random.randn(hidden_size,output_size) * init_std
        self.params['b2'] = np.zeros(output_size)

    def predict(self,x):
        tmp1 = np.dot(x,self.params['w1']) + self.params['b1']
        z1 = sigmoid(tmp1)

        tmp2 = np.dot(z1,self.params['w2']) + self.params['b2']
        z2 = softmax(tmp2)
        return z2
    
    def loss(self,x,t):
        return cross_entropy_error(self.predict(x),t)
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y,t = np.argmax(y),np.argmax(t)
        return np.sum(y==t) / float(t.shape[0])

    def numerical_grads(self,x,t):
        f = lambda w : self.loss(x,t)
        grads = {}
        grads['w1'],grads['b1'] = numerical_diff(f,self.params['w1']), \
                                  numerical_diff(f,self.params['b1'])
        grads['w2'],grads['b2'] = numerical_diff(f,self.params['w2']), \
                                  numerical_diff(f,self.params['b2'])
        return grads

if __name__ == '__main__':
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(
    net.params['w1'].shape,
    net.params['b1'].shape,
    net.params['w2'].shape,  
    net.params['b2'].shape,
    )
    '''
    x = np.random.rand(100, 784) 
    y = net.predict(x)
    print(y.shape)
    '''

    x,x_label,t,t_label = load_minist()

    #print(x.shape,x_label.shape)
    #print(t.shape,t_label.shape)
    #print(x[0])
    
    # SGD
    iter_num = 1000
    trian_size = x.shape[0]
    batch_size = 100
    lr = 0.1
    loss_list = []
    
    for i in tqdm(range(iter_num)):
        idx_set = np.random.choice(trian_size,batch_size)
        data = x[idx_set]
        label = x_label[idx_set]

        grads = net.numerical_grads(data,label)
        for key in ('w1','b1','w2','b2'):
            net.params[key] -= lr*grads[key]
        loss_list.append(net.loss(data,label))
        
    import pylab as plt
    plt.plot(np.arange(iter_num),loss_list)
    plt.show()
    