import numpy as np


def exponential_decay(t):
    t=np.abs(t)
    k = 0.008 # With this constant, a 300 frame difference (10 seconds at 30fps) adds just 0.1 to the cm cell
    return np.exp(-k*t)

def inverse_decay(t):
    t=np.abs(t)
    return 1/t

def linear_decay(t):
    t=np.abs(t)
    k = 0.0003 # With this constant, a 300 frame difference (10 seconds at 30fps) adds just 0.09 to the cm cell
    return max(0, 1 - k*t) 

def log_decay(t):
    t=np.abs(t)

    return max(0, 1-np.log(t))

def sigmoid_decay(t):
    np.sigmoid(t)