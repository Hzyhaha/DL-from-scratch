import numpy as np
from diffraction import numerical_diff

def gradient_descent(f,init_x,lr=0.01,num_step=100):
    x = init_x

    for i in range(num_step):
        grad = numerical_diff(f,x)
        x -= lr*grad
    
    return x

if __name__ == '__main__':
    from diffraction import func1
    init_x = np.array([-3.0,4.0])
    min_x = gradient_descent(func1,init_x,lr=0.1)
    #注意书中改了lr
    print(min_x,func1(min_x))