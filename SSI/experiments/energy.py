import math


def getSampleDimensionOut(dim, opts, vars): # redundant
    return dim


def getSampleBytesOut(bytes, opts, vars): # redundant
    return bytes


def getSampleTypeOut(type, types, opts, vars): # redundant
    return type


def transform_enter(sin, sout, sxtras, board, opts, vars): # redundant
    pass


def transform(info, sin, sout, sxtras, board, opts, vars):   

    # Set the output to 0
    for d in range(sin.dim):
        sout[d] = 0

    # Square the value in input and sum the dimensions
    for n in range(sin.num):
        for d in range(sin.dim):
            val = sin[n,d]
            sout[d] += val*val

    # Take the square root of the square sum of each dimension divided by the number of dimension
    for d in range(sin.dim):
        sout[d] = math.sqrt(sout[d] / sin.num)   


def transform_flush(sin, sout, sxtras, board, opts, vars):  # redundant 
    pass