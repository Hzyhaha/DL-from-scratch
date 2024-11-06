import numpy as np
import matplotlib
from matplotlib import pylab as plt

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

def ReLU(x):
    return np.maximum(0,x)
    #max只能是一个数组中找出最大,这里是两个数组对应位置取最大，用maximun

def ReLU2(x):
    return x*step_function(x)

if __name__ == '__main__':

    
    draw_func(step_function)
    draw_func(sigmoid)
    draw_func(ReLU)
    # draw_func(ReLU2)
    plt.legend(['step_function','sigmoid','ReLU'])
    plt.grid(True)
    plt.show()
    
    
