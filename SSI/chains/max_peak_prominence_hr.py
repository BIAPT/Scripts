import os
import sys
import random
import math

from features import find_max_peak_prominence
sys.path.append(os.environ.get('PYTHONPATH', ''))


def getOptions(opts, vars):
    return None

def getSampleDimensionOut(dim, opts, vars):
    return 1


def getSampleBytesOut(bytes, opts, vars): # redundant
    return bytes


def getSampleTypeOut(type, types, opts, vars): # redundant
    return type


def transform_enter(sin, sout, sxtras, board, opts, vars): # redundant
    pass


def transform(info, sin, sout, sxtras, board, opts, vars):   
    sout = find_max_peak_prominence(sin)

def transform_flush(sin, sout, sxtras, board, opts, vars):  # redundant 
    pass