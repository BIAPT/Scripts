# General
import random

# Data manipulation
import scipy.io
import numpy as np

# Make a mask to separate training from test participant
def make_mask(I,target):
    test_mask = np.where(I == target)
    mask = np.ones(len(I), np.bool)
    mask[test_mask] = 0
    train_mask = np.where(mask == 1)
    return (train_mask[0],test_mask[0])

def load_data():
    data = scipy.io.loadmat('data/X_aec.mat')
    X_aec = np.array(data['X'])
    data = scipy.io.loadmat('data/X_pli.mat')
    X_pli = np.array(data['X'])
    data = scipy.io.loadmat('data/Y.mat')
    y = np.array(data['Y'])
    y = y[0]
    data = scipy.io.loadmat('data/I.mat')
    I = np.array(data['I'])
    I = I[0]
    return (X_pli,X_aec,y,I)
