import numpy as np
from loss_func import cross_entropy_error
from diffraction import numerical_diff
from grad_desc import gradient_descent

def softmax(x):
    C = np.max(x)
    tmp = np.exp(x-C)
    return tmp/np.sum(tmp)

class simple_net:
    def __init__(self) -> None:
        self.weight = np.random.rand(2,3)

    def predict(self,x):
        return x@self.weight
    
    def loss(self,x,t):
        z = self.predict(x)
        return cross_entropy_error(softmax(z),t)
    



if __name__ == '__main__':
    net = simple_net()
    print(net.weight)
    x = np.array([0.6, 0.9])
    print(net.predict(x))
    t = np.array([0.0,1.0,0.0])
    print(net.loss(x,t))
    def f(w):
        return net.loss(x,t)
    dw = numerical_diff(f,net.weight)
    print(dw)

    gradient_descent(f,net.weight,lr=0.01,num_step=1000)

    print('final loss: ',net.loss(x,t),'probability: ',softmax(net.predict(x)))