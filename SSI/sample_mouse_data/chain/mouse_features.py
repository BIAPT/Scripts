import numpy as np
import os
import sys
sys.path.append('.')

import features as feat

def print_stream(s):

    print("> object type = %s" % type(s))
    print("> value type  = %s" % s.type())
    print("> shape       = %d x %d" % s.shape())    
    print("> len         = %d" % s.len) 
    print("> num         = %d" % s.num) 
    print("> sr          = %f" % s.sr) 
    print("> tot         = %d" % s.tot) 
    print("> byte        = %d" % s.byte) 


def getOptions(opts, vars):
    opts['features'] = {
        'min': True,
        'max' : True,
        'avg' : True,
        'dist' : True,
        'var' : True,
        'std' : True
    }

    print('get options: {}'.format(opts))


def getSampleDimensionOut(dim, opts, vars):
    d = 0
    print(opts)
    features = opts['features']
    
    if features['min']:
        d += 2

    if features['max']:
        d += 2

    if features['avg']:
        d += 2

    if features['dist']:
        d += 1

    if features['var']:
        d += 1

    if features['std']:
        d += 1

    return d


def getSampleBytesOut(bytes, opts, vars):
    return 4


def getSampleTypeOut(type, types, opts, vars):
    return types.FLOAT


def transform_enter(sin, sout, sxtra, board, opts, vars):
    pass


def transform(info, sin, sout, sxtra, board, opts, vars):   
    # sin is a py-stream
    sample_np = np.asarray(sin)

    features = feat.calc_feature_from_sample(sample_np, opts)
    
    for i in range(0,sout.dim):
        sout[i] = features[i]

def transform_flush(sin, sout, sxtra, board, opts, vars):   
    pass
