import numpy as np

def mean_squared_error(y,t):
    '''
    :param y: the output of the NN
    :param t: the label of the input
    '''
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    '''
    :param y: the output of the NN
    :param t: the label of the input
    '''
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

def minibatch_cross_entropy_error(y,t):
    '''
    :param y: the output of the NN
    :param t: the label (one-hot) of the input 
    '''
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1,-1)
        t = t.reshape(1,-1)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+delta))/batch_size