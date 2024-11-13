import numpy as np
import sys 
sys.path.append('./common')
from diffraction import numerical_diff
from collections import OrderedDict
from layers import Affine,Softmax_Loss_Layer,ReLU
class TwoLayerNet :
    def __init__(self,
                 input_size,hidden_size,output_size,
                 init_std = 0.01) -> None:
        self.params = {}
        self.params['w1'] = np.random.randn(input_size,hidden_size) * init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = np.random.randn(hidden_size,output_size) * init_std
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()

        self.layers['affine1'] = \
        Affine(self.params['w1'],self.params['b1'])
        self.layers['ReLU1'] = ReLU()

        self.layers['affine2'] = \
        Affine(self.params['w2'],self.params['b2'])

        self.lastlayer = Softmax_Loss_Layer()

    def predict(self,x):
        for value in self.layers.values():
            x = value.forward(x)
        return x

    
    def loss(self,x,t):
        z = self.predict(x)
        loss = self.lastlayer.forward(z,t)
        return loss
    
    def accuracy(self,x,t):
        y = self.predict(x)
        # 这里对数据维度有要求，并且输入都是批量数据
        y = np.argmax(y,axis=1)
        if t.ndim != 1: 
            t = np.argmax(t,axis=1)
        return np.sum(y==t) / float(t.shape[0])

    def numerical_grads(self,x,t):
        f = lambda w : self.loss(x,t)
        grads = {}
        grads['w1'],grads['b1'] = numerical_diff(f,self.params['w1']), \
                                  numerical_diff(f,self.params['b1'])
        grads['w2'],grads['b2'] = numerical_diff(f,self.params['w2']), \
                                  numerical_diff(f,self.params['b2'])
        return grads #返回梯度字典
    
    def gradient(self,x,t):
        #前向传播，记忆必要数据
        self.loss(x,t)
        
        # 反向传播
        dout = self.lastlayer.backward()
        for value in self.layers.values():
            dout = value.backward(dout)
        grads = {}
        grads['w1'],grads['b1'] = \
        self.layers['Affine1'].dw, self.layers['Affine1'].db

        grads['w2'],grads['b2'] = \
        self.layers['Affine2'].dw, self.layers['Affine2'].db
        return grads #返回梯度字典

if __name__ == '__main__':
    from load_data import load_minist
    from tqdm import tqdm
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(
    net.params['w1'].shape,
    net.params['b1'].shape,
    net.params['w2'].shape,  
    net.params['b2'].shape,
    )

    x,x_label,t,t_label = load_minist()
  
    # SGD
    iter_num = 100
    trian_size = x.shape[0]
    batch_size = 100
    lr = 0.1
    loss_list = []
    
    for i in tqdm(range(iter_num)):
        idx_set = np.random.choice(trian_size,batch_size)
        data = x[idx_set]
        label = x_label[idx_set]

        grads = net.gradient(data,label)
        for key in ('w1','b1','w2','b2'):
            net.params[key] -= lr*grads[key]
        loss_list.append(net.loss(data,label))
        
    import pylab as plt
    plt.plot(np.arange(iter_num),loss_list)
    plt.show()