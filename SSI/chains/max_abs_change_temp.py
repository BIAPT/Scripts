import os
import sys
import random
import math
import numpy as np
from features import find_max_abs_change
sys.path.append(os.environ.get('PYTHONPATH', ''))


def getOptions(opts, vars):
    opts['window_size'] = 10 # in sec

def getSampleDimensionOut(dim, opts, vars):
    return 1


def getSampleBytesOut(bytes, opts, vars): # redundant
    return bytes


def getSampleTypeOut(type, types, opts, vars): # redundant
    return type


def transform_enter(sin, sout, sxtras, board, opts, vars): # redundant
    pass


def transform(info, sin, sout, sxtras, board, opts, vars):   
    window_size = opts['window_size']
    x = np.linspace(0,window_size,len(sin))
    sout = find_max_abs_change(x,sin)


def transform_flush(sin, sout, sxtras, board, opts, vars):  # redundant 
    pass