import os
import sys
import random
import math
sys.path.append(os.environ.get('PYTHONPATH', ''))

def smoothing_factor(cutoff):
    r = 2 * math.pi * cutoff
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

def one_euro(input_data, output_data, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
    '''
    Filter recommended by Dr. Florian Grond for the filtering of signals in realtime.
    '''

    # Parameters initalization
    min_cutoff = float(min_cutoff)
    beta = float(beta)
    d_cutoff = float(d_cutoff)
 
    # Running this algorithm on a consecutive
    output_data[0] = input_data[0]
    x_previous = input_data[0]
    dx_previous = 0.0
    for i in range(1,input_data.num):

        # Get the data
        x = input_data[i]

        # The filtered derivative of the signal.
        a_d = smoothing_factor(d_cutoff)
        dx = (x - x_previous)
        dx_hat = exponential_smoothing(a_d, dx, dx_previous)

        # The filtered signal.
        cutoff = min_cutoff + beta * abs(dx_hat)    
        a = smoothing_factor(cutoff)
        x_hat = exponential_smoothing(a, x, x_previous)   
        
        # Memorize the previous values.
        x_previous = x_hat
        dx_previous = dx_hat

        output_data[i] = x_hat

def getOptions(opts, vars):
    opts['min_cutoff'] = 50.0
    opts['beta'] = 4.0
    opts['d_cutoff'] = 1.0

def getSampleDimensionOut(dim, opts, vars):
    return 1


def getSampleBytesOut(bytes, opts, vars): # redundant
    return bytes


def getSampleTypeOut(type, types, opts, vars): # redundant
    return type


def transform_enter(sin, sout, sxtras, board, opts, vars): # redundant
    pass


def transform(info, sin, sout, sxtras, board, opts, vars):   
    min_cutoff = opts['min_cutoff']
    beta = opts['beta']
    d_cutoff = opts['d_cutoff']
    one_euro(sin, sout, min_cutoff, beta, d_cutoff)


def transform_flush(sin, sout, sxtras, board, opts, vars):  # redundant 
    pass