import math

'''
    This code was taken from: https://github.com/jaantollander/OneEuroFilter
    and written by Jaan Tollander de Balsch 
    It was edit and refactored by Yacine Mahdid in 2019-11-06, I removed the Class OneEuroFilter
    and replaced it with a function that can be called directly with an array of data points
'''

def smoothing_factor(cutoff):
    r = 2 * math.pi * cutoff
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

def one_euro(input_data, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
    '''
    Filter recommended by Dr. Florian Grond for the filtering of signals in realtime.
    '''

    # Parameters initalization
    output_data = []
    min_cutoff = float(min_cutoff)
    beta = float(beta)
    d_cutoff = float(d_cutoff)
 
    # Running this algorithm on a consecutive
    x_previous = input_data[0]
    dx_previous = 0
    for i in range(1,len(input_data)):

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

    return output_data

