import numpy as np 
import pandas as pd

def wpli(data, number_surrogates, p_value):
    '''
    Weighted Phase Lag Index functions operating on C*P matrix
    where C is the number of channels and P is the number of data points
    In the matrix.
    The number of surrogates are used along with the p value to test
    for significance using phase shuffled surrogate wPLI matrix with
    similar properties than the calculated wPLI matrix
    '''

    # Variables Initalization
    number_channels = np.size(data,1) # Get the size of the first dimension (TODO: Check)
    surrogates_wpli = np.zeros(number_surrogates,number_channels,number_channels)
    data = data.T() # TODO: Double check this 

    # Calculate the wPLI
    uncorrected_wpli = calculate_uncorrected_wpli(data)
    # TODO: Fix the NaN problem here

    # Generate the surrogates
    for i in range(0,number_surrogates):
        surrogates_wpli[i,:,:] = calculate_surrogate_wpli(data)

    # do a p test and create the corrected wPLI
    corrected_wpli = zeros(number_channels, number_channels)
    for channel_i in range(0,number_channels):
        for channel_j in range(0, number_channels):
            test = surrogates_wpli[:,channel_i,channel_j]
            test_median = np.median(test)
            # get a p value
            p = -1
            if p < p_value and uncorrected_wpli(channel_i, channel_j) - test_median > 0:
                corrected_wpli[channel_i, channel_j] = uncorrected_wpli[channel_i, channel_j] - test_median

    return corrected_wpli

def calculate_uncorrected_wpli(data):
    '''
    Helper function of wpli to calculate uncorrected weighted phase lag index on a C*P matrix
    (See wpli for more information)
    '''
    print("Another dummy function")

# Check if there is a way to condense the code here to use the calculate uncorrected wpli above
def calculate_surrogate_wpli(data):
    '''
    Helper function of wpli to calculate phase shuffled weighted phase lag index on a C*P matrix
    (See wpli for more information)
    '''
    print("Third dummy function")
