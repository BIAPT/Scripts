import os
import sys
import random
import math
#import numpy  

# Signal processing and data analysis imports
#from scipy.signal import find_peaks, peak_prominences

#import scipy

# Features for HearthRate
# Will calculate the maximum peak prominence for the data given
def find_max_peak_prominence(data):
    peaks, _ = find_peaks(data)
    prominences = peak_prominences(data, peaks)[0]
    return np.max(prominences)

def getSampleDimensionOut(dim, opts, vars): # redundant
    return dim


def getSampleBytesOut(bytes, opts, vars): # redundant
    return bytes


def getSampleTypeOut(type, types, opts, vars): # redundant
    return type


def transform_enter(sin, sout, sxtras, board, opts, vars): # redundant
    pass


def transform(info, sin, sout, sxtras, board, opts, vars): 
    print(sin)
    sout = sin


def transform_flush(sin, sout, sxtras, board, opts, vars):  # redundant 
    pass