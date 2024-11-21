import numpy as np
import pylab as plt

def step_function(x):
    y = x>0
    return y.astype(int)
'''
x = np.array([0,1,2,0.2,-2])
print(step_function(x))
'''


def sigmoid(x):
    return 1/(1+np.exp(-x))


def draw_func(func):
    x = np.arange(-5,5,0.1)
    y = func(x)
    plt.plot(x,y)


plt.figure()

def ReLU(x):
    return np.maximum(0,x)
#max只能是一个数组中找出最大,这里是两个数组对应位置取最大，用maximun

def ReLU2(x):
    return x*step_function(x)

def softmax(x):
    # numpy中二维矩阵减一维数组，默认一维是行向量然后延拓
    if x.ndim == 2:
        
        x = x - np.max(x, axis=1).reshape(-1,1)
        y = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1,1)
        return y

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


if __name__ == '__main__':
    draw_func(step_function)
    draw_func(sigmoid)
    draw_func(ReLU)
    plt.show()
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)