import numpy as np

# 似乎是单输入单输出
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


    
if __name__ == '__main__':
    testlayer = ReLU()
    x = np.array([3.0])
    y = testlayer.forward(x)
    dx = testlayer.backward(np.array([1.0]))
    print(y);print(dx)

    testlayer2 = Sigmoid()
    x = np.linspace(-5.0,5.0,100)
    y = testlayer2.forward(x)
    import pylab as plt
    plt.plot(x,y)
    plt.show()