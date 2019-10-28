# General Import
import os
from math import log
# General Data science Import
import numpy as np
import scipy.io
# Current Topic Specific Import
#import mne # MNE seems cool, but its a bit too low level for what I need right now
from lempel_ziv_complexity import lempel_ziv_complexity
import apci

# Visualization
import matplotlib.pyplot as plt

# Helper functions
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
pci = apci.alpha_power_apci(mean_baseline, std_baseline, real_matrices)
print("Propofol Induced APCI for participant " + participant_label + " = " + str(pci))
pci = apci.alpha_power_apci(mean_baseline, std_baseline, control_matrices)
print("Control APCI for participant " + participant_label + " = " + str(pci))

