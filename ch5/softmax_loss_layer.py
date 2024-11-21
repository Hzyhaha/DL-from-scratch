import numpy as np
import sys
sys.path.append('.')
from hzy_common.activation_func import softmax
from hzy_common.loss_func import cross_entropy_error

class Softmax_Loss_Layer:
    def __init__(self) -> None:
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t)*dout/batch_size

if __name__ == '__main__':
    x = np.array([1,2,1])
    t = np.array([0,1,0])
    testlayer = Softmax_Loss_Layer()
    testlayer.forward(x,t)
    print(testlayer.y,testlayer.loss,testlayer.backward())