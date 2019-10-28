import os
import sys
import random
sys.path.append(os.environ.get('PYTHONPATH', ''))

# Exponential Decay Smoothing Filter
#increase alpha for more filtering: uses more past data to compute an average
def exponential_decay(input_data, output_data, alpha):
    '''
    Exponential Decay Smoothing Filter
    Increase alpha for more filtering: uses more past data to compute an average
    '''
    print(alpha)
    output_data[0] = input_data[0]
    for i in range(0,input_data.num-1):
        output_data[i+1] = (output_data[i,0]*alpha) + (input_data[i+1,0] * (1-alpha))

def getOptions(opts, vars):
    opts['alpha'] = 0.5

def getSampleDimensionOut(dim, opts, vars):
    return 1


def getSampleBytesOut(bytes, opts, vars): # redundant
    return bytes


def getSampleTypeOut(type, types, opts, vars): # redundant
    return type


def transform_enter(sin, sout, sxtras, board, opts, vars): # redundant
    pass


def transform(info, sin, sout, sxtras, board, opts, vars):   
    alpha = opts['alpha']
    exponential_decay(sin, sout, alpha)


def transform_flush(sin, sout, sxtras, board, opts, vars):  # redundant 
    pass