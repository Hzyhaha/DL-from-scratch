import numpy as np

class SGD():
    def __init__(self,lr=0.01) -> None:
        self.lr = lr

    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum():
    def __init__(self,lr=0.01,momentum=0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def update(self,params,grads):
        if self.velocity == None:
            self.velocity = {}
            for key,val in params.items():
                self.velocity[key] = np.zeros_like(val)

        for key in params.keys():
            # 梯度是相对于损失函数的，是正值代表增大的方向，故取反
            self.velocity[key] = self.momentum*self.velocity[key] \
                                 -self.lr * grads[key]
            # 这里的速度是下降到最低点的速度，所以用加号
            params[key] += self.velocity[key]