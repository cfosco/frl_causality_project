import numpy as np


def exponential_decay(t):
    t=np.abs(t)
    k = 0.15 # With this constant, a 300 frame difference (10 seconds at 30fps) adds just 0.1 to the cm cell
    return np.exp(-k*t)

def exp_decay_idx(t):
    t=np.abs(t)
    k = 0.08
    return np.exp(-k*t)

def inverse_decay(t):
    t=np.abs(t)
    k=1
    return 1/(k*(t+1))

def linear_decay(t):
    t=np.abs(t)
    k = 0.03 # With this constant, a 300 frame difference (10 seconds at 30fps) adds just 0.09 to the cm cell
    return np.maximum(np.zeros(len(t)), 1 - k*t) 

def log_decay(t):
    t=np.abs(t)

    return np.maximum(np.zeros(len(t)), 1-np.log(t))

def sigmoid_decay(t):
    t=np.abs(t)
    k=1
    d=5
    return 1 / (1 + np.exp(k*(t-d)))