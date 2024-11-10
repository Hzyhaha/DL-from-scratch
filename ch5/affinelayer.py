import numpy as np

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
        self.b = np.sum(dout,axis=0)
        dx = dout@self.w.T
        return dx

if __name__ == '__main__':
    x = np.array([[1.0,2.0],[-1.0,3.0]])
    w = np.array([[1.0],[-1.0]])
    b = np.array([[1.0],[1.0]])
    testlayer = Affine(w,b)
    out = testlayer.forward(x)
    print(out)
    print(testlayer.backward(np.array([[1.0],[1.0]])))
