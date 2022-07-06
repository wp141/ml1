import numpy as np

def ReLU(z):
    return np.maximum(0, z)

def derivative_ReLU(z):
    return z > 0

def softmax(x):
    softmax = np.exp(x) / np.sum(np.exp(x)) 
    return softmax

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1 
    return one_hot_y.T