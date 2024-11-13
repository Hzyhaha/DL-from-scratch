import numpy as np

from activation_func import softmax
from loss_func import minibatch_cross_entropy_error as cross_entropy_error

'''
大原则是反向传播时需要哪些 就需要记忆为self的属性

反向传播返回的梯度是输入数据的梯度
权重的梯度需要记忆为self的属性

out: 前向传播的输出流
dout: 反向传播的输入流 也即是上一次的输出流

forward方法只要x 最后一层才需要x和t一起
backward由于已经存储所需要计算的 故只要传dout
但是要注意，dout是关于数据的梯度，而非关于参数的梯度，参数梯度已经被存储了
'''


class MulLayer:
    def __init__(self) -> None:
        # 直接pass的话，必须forward才有这个属性
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        return x*y
    
    def backward(self,dout):
        return self.y*dout,self.x*dout

class AddLayer():
    def __init__(self) -> None:
        pass

    def forward(self,x,y):
        return x+y
    
    def backward(self,out):
        return out,out
    

class ReLU :
    def __init__(self) -> None:
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self,dout):
        dout[self.mask] = 0
        return dout
    
class Sigmoid:
    def __init__(self) -> None:
        self.out = None #等下计算梯度的时候用到哪个，就把哪个存起来
    
    def forward(self,x):
        out = 1.0 / (1.0+np.exp(-x))
        self.out = out
        return out
    
    def backward(self,dout):
        return dout * self.out * (1.0-self.out)
    
class Affine:
    def __init__(self,w,b) -> None:
        self.x = None
        self.w = w
        self.b = b

        self.dw = None
        self.db = None

    def forward(self,x):
        self.x = x
        return np.dot(x,self.w) + self.b
    
    def backward(self,dout):
        self.dw = self.x.T@dout
        self.db = np.sum(dout,axis=0)
        dx = dout @ (self.w.T)
        return dx
    
class Softmax_Loss_Layer:
    # 以得分为输入，以概率向量为中间输出，最终以交叉熵损失为输出
    def __init__(self) -> None:
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss # return the Loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t)*dout/batch_size