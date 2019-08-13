# General Import
import os
from math import log
# General Data science Import
import numpy as np
import scipy.io
# Current Topic Specific Import
#import mne # MNE seems cool, but its a bit too low level for what I need right now
from lempel_ziv_complexity import lempel_ziv_complexity

# Visualization
import matplotlib.pyplot as plt

#######################################################
# TODO: bundle these helper function in a package, because it will be used by other scripts
#######################################################

# Helper functions

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
def awareness_perturbation_complexity_index(matrix):
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

    apci = awareness_perturbation_complexity_index(binary_stream)
    return apci
    

def average_frequency(matrix):
    return np.mean(matrix, axis=1)

def load_data(path):
    data = scipy.io.loadmat(path)
    return average_frequency(np.array(data['y']))

# Variable Intializatin
participant_label = "MDAF03"

# Load all the data of interest
baseline = load_data('data/alpha_baseline.mat')
induction = load_data('data/alpha_induction.mat')
emergence_first_5 = load_data('data/alpha_emergence_first_5.mat')
emergence_last_5 = load_data('data/alpha_emergence_last_5.mat')
eyes_closed_5 = load_data('data/alpha_eyes_closed_5.mat')
eyes_closed_6 = load_data('data/alpha_eyes_closed_6.mat')
eyes_closed_7 = load_data('data/alpha_eyes_closed_7.mat')
recovery = load_data('data/alpha_recovery.mat')

# Create the data matrices
real_matrices = [baseline, induction, emergence_first_5, emergence_last_5, recovery]
control_matrices = [baseline, eyes_closed_5, eyes_closed_6, eyes_closed_7, recovery]

# gather the statistics for the baseline
mean_baseline = np.mean(baseline)
std_baseline = np.std(baseline)
print("Mean of baseline: " + str(mean_baseline))
print("Std of baseline: " + str(std_baseline))

# Calculate the APCI
apci = alpha_power_apci(mean_baseline, std_baseline, real_matrices)
print("Propofol Induced APCI for participant " + participant_label + " = " + str(apci))
apci = alpha_power_apci(mean_baseline, std_baseline, control_matrices)
print("Control APCI for participant " + participant_label + " = " + str(apci))

