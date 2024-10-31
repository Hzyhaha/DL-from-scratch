import numpy as np

def old_numerical_diff(f,x):
    '''
    :param f: Function
    :param x: the point where you need to diff
    
    '''
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)

def numerical_diff(f,x:np.ndarray):

    h = 1e-4
    grad = np.zeros_like(x)
    
    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        forward = f(x)

        x[i] = tmp - h
        back = f(x)

        x[i] = tmp
        grad[i] = (forward - back)/2/h
    return grad

def func1(x):
    return x[0]**2+x[1]**2

if __name__=='__main__':
    print(numerical_diff(func1,np.array([3.0,0.0])))
    # 整数会出现精度误差