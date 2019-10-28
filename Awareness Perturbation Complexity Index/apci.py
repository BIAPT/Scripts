import numpy as np
from math import log
from lempel_ziv_complexity import lempel_ziv_complexity

# Here we assume that the matrix is made like this Matrix(X,T) where X is the features and T are the time point
# and that the data is arranged like this Matrix(row,column)
def matrix_to_string(matrix):
    if(type(matrix) == list):
        array = matrix
    else:
        num_row, num_col = matrix.shape
        array = matrix.flatten("F")
    string_array = ''.join(str(n) for n in array)
    return string_array

# This function accepts a matrix(row,col) where row = features X and col = time T
def calculate_apci(matrix):
    # Here we conver the matrix into a stream by concatenating column wise
    stream = matrix_to_string(matrix)
    
    # Variable initialization
    length_stream = len(stream)
    p = stream.count("1")/length_stream
    source_entropy = -p*log(p, 2) - (1-p)*log((1-p), 2)
    complexity = lempel_ziv_complexity(stream) # Here we could use the cython version for speedup
    
    # PCI formula
    pci = complexity*(log(length_stream, 2)/(length_stream*source_entropy))
    return pci

# Input here are the 5 stages of interest
# 1) Eyes closed 1 (Baseline)
# 2) Induction first 5 minute
# 3) Emergence first 5 minute
# 4) Emergence last 5 minute
# 5) Eyes closed 8 (Recovery)
# What we will be looking for here is to caluclate the mean and standard deviation of the baseline
# Then we will check which values are Xstandard deviation away from the baseline.
# If they are we put a 1 if not we will put a 0
def alpha_power_apci(mean_baseline, std_baseline, matrices):
    # Variable initialization
    a = 2 # multiplier for the standard deviation
    min_power = mean_baseline - a*std_baseline
    max_power = mean_baseline + a*std_baseline

    binary_stream = []
    counter = 0
    for matrix in matrices:
        # Here we threshold each value and we put it in a list
        for value in matrix:
            if(value > max_power or value < min_power):
                binary_stream.append(1)
            else:
                binary_stream.append(0)
            counter = counter + 1
        print(counter)

    print("Number of 1s = " + str(binary_stream.count(1)))
    print("Total number of element = " + str(counter)) 
    plt.plot(binary_stream)
    plt.show()

    score = calculate_apci(binary_stream)
    return score

# Maps thresholding based upon the baseline for all the other maps
def threshold_topographic_maps(baseline_maps, test_maps, threshold):
    # We create a distribution using the values from all the baseline maps
    (number_repetitions, number_channels) = baseline_maps.shape
        # Make a min and max threshold value
    min_index = int(number_repetitions * (threshold/2)) # 0.025 if p < 0.05
    max_index = int(number_repetitions * ( 1- (threshold/2))) # 0.975 if p < 0.05

    significant_limit = np.zeros([number_channels, 2]) # i = 0 is min i = 1 is max
    # Sort the values from smallest to largest and populate the significance limit
    for i in range(0, number_channels):
        baseline_maps[:,i] = np.sort(baseline_maps[:,i])
        # Geting the right limits for the distribution 

        significant_limit[i,0] = baseline_maps[min_index,i]
        significant_limit[i,1] = baseline_maps[max_index,i]
    
    # Set the whole maps to 0 like in the PCI paper
    baseline_maps = np.zeros([number_repetitions, number_channels])

    (number_repetitions, number_channels) = test_maps.shape
    # iterate over the test_maps and threshold them using the significan_litmit
    for m in range(0, number_repetitions):
        for i in range(0, number_channels):
            # Check if this value is smaller or bigger than the distribution limit we created
            if(test_maps[m,i] <= significant_limit[i,0] or test_maps[m,i] >= significant_limit[i,1]):
                test_maps[m,i] = 1
            else:
                test_maps[m,i] = 0
    
    binary_topographic_maps = np.concatenate((baseline_maps, test_maps)) 
    return binary_topographic_maps


    

# Maps here are C length vector and there are N of them
# Which means we have N*C Matrix
# Here the maps are binary topographical maps at the source level
def topographic_alpha_power_apci(topographic_maps):
    (number_maps, number_channels) = topographic_maps.shape    
    binary_stream = []
    # Iterate through the rows (i) and the maps (m) and repeat columnwise
    for i in range(0, number_channels):
        for m in range(0, number_maps):
            binary_stream.append(topographic_maps[m, i])
    
    score = calculate_apci(binary_stream)
    return score
        
