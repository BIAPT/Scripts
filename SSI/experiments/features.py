# Visualization imports
import matplotlib.pyplot as plt

# Signal processing and data analysis imports
from scipy.signal import find_peaks, peak_prominences
import numpy as np 
import scipy

# Features for HearthRate
# Will calculate the maximum peak prominence for the data given
def find_max_peak_prominence(data):
    peaks, _ = find_peaks(data)
    prominences = peak_prominences(data, peaks)[0]
    return np.max(prominences)

# Helper function for finding derivative numerically
def find_derivative(x,y):
    dy = np.zeros(y.shape,np.float)
    dy[0:-1] = np.diff(y)/np.diff(x)
    dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    return dy

# Feature for temperature
def find_max_abs_change(x,y):
    dy = find_derivative(x,y)
    dy = np.abs(dy)
    return np.max(dy)

# Feature for eda
def find_max_change(x,y):
    dy = find_derivative(x,y)
    return np.max(dy)