import os
import sys
import random
sys.path.append(os.environ.get('PYTHONPATH', ''))

def myFilter(input_data, output_data):
    for n in range(input_data.num):
        output_data[n] = 0
        for d in range(input_data.dim):
            output_data[n] += input_data[n,d]*(random.random())

def myOtherFilter(input_data, output_data):
    for n in range(input_data.num):
        for d in range(input_data.dim):
            output_data[n] += input_data[n,d]*(random.random())            

def getSampleDimensionOut(dim, opts, vars):
    return 1


def getSampleBytesOut(bytes, opts, vars): # redundant
    return bytes


def getSampleTypeOut(type, types, opts, vars): # redundant
    return type


def transform_enter(sin, sout, sxtras, board, opts, vars): # redundant
    pass


def transform(info, sin, sout, sxtras, board, opts, vars):   
    myFilter(sin,sout)
    myOtherFilter(sout,sout)


def transform_flush(sin, sout, sxtras, board, opts, vars):  # redundant 
    pass
