import math

# Data manipulation imports
import csaps 
import numpy as np 

# Visualization imports
import matplotlib.pyplot as plt

# Filters
def cubic_spline_smoothing(x, y, p=0.01, new_sampling_rate=2):
    '''
        We are using the csaps python package which is a port of the matlab
        using this port https://pypi.org/project/csaps/
        This is used for HR in the original movingwith paper

        - p is the smoothing parameter, we can decrease it for more smoothing
        however 0.001 is what we used previously
        
        - new_sampling_rate is how much more data points (in Hz) we want to interpolate the gaps

    '''
    filter = csaps.UnivariateCubicSmoothingSpline(x, y, smooth=p)
    
    # Calculate an approximative sampling rate
    #current_sampling_rate = (x[-1] - x[0]) / (1000*len(x)) # The time is in milliseconds so we put them in seconds
    
    # get the ratio of augmentation
    #ratio = new_sampling_rate / current_sampling_rate

    new_len = new_sampling_rate * ((x[-1] - x[0])/1000)
    # Augment the data to match the wanted sampling rate
    filt_x = np.linspace(x[0], x[-1], new_len)
    filt_y = filter(filt_x)

    return (filt_x, filt_y)

def exponential_decay(y, alpha=0.95):
    '''
    Exponential Decay Smoothing Filter
    This is used for Temperature in the original movingwith paper
    Increase alpha for more filtering: uses more past data to compute an average    
    '''

    filt_y = y.copy()
    for i in range(0,len(y)-1):
        filt_y[i+1] = (filt_y[i]*alpha) + (filt_y[i+1] * (1-alpha))
        
    return filt_y

# One Euro filtering code
def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    '''
    This code was taken from: https://github.com/jaantollander/OneEuroFilter
    and written by Jaan Tollander de Balsch 
    We first need to init the OneEuroFilter with the base parameters and one data point.
    Then we repeatedly call OneEuroFilter(t,x) on the data points to filter
    i.e.
    oneEuro = OneEuroFilter(t0,x0,min_cutoff=50.0,beta=4.0)
    ...
    x_filt = oneEuro(ti,xi)
    '''
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

def one_euro(x,y):
    filter = OneEuroFilter(x[0], y[0], min_cutoff=50.0, beta=4.0)
    filt_y = [y[0]]
    for i in range(1,len(x)):
        filt_y.append(filter(x[i],y[i]))

    return filt_y

# Averaging
def window_average(timestamps, data, window_size=0.5):  
    '''
        window averaging with a jumping window (default is 0.5 seconds)
        the bigger the size of the window the smaller the new sample rate will be
        if we choose a window of 0.5 it will be 2Hz
    '''

    # convert the window size to be in milliseconds
    window_size = int(window_size*1000)

    # setting up the arrays and variables
    avg_timestamps = []
    avg_data = []
    current_i = 0

    # Iterating through each windows
    for start_t in range(timestamps[0], timestamps[-1], window_size):
        
        curr_data = []
        current_t = timestamps[current_i]
        end_t = start_t + window_size

        # Appending all the data within that window
        while current_t < end_t:
            curr_data.append(data[current_i])
            current_i = current_i + 1
            if current_i >= len(timestamps):
                break
            current_t = timestamps[current_i]
        
        # If there is nothing in this time window we don't add it
        if len(curr_data) != 0:
            avg_data.append(np.mean(curr_data))

            # Adding the timestamp to be the middle of the average window
            avg_timestamps.append(start_t + (window_size/2))

    return (avg_timestamps, avg_data)

# Pre processing function that will apply the pre-processing technique
# based on what analysis technique we are working with.
def pre_process(timestamps, data, data_type):
    print("Preprocessing: " + data_type)

    # Window averaging at 2Hz (0.5seconds)
    (avg_timestamps, avg_data) = window_average(timestamps,data)

    # Pre-processing (if you want to tweak them change them here!)
    if data_type == "EDA":
        filt_data = one_euro(avg_timestamps, avg_data)
        #plt.plot( timestamps, data, 'o', avg_timestamps, filt_data, '-')
        #plt.show()
    elif data_type == "TEMP":
        filt_data = exponential_decay(avg_data)
        #plt.plot( timestamps, data, 'o', avg_timestamps, filt_data, '-')
        #plt.show()
    elif data_type == "HR":
        (filt_timestamps, filt_data) = cubic_spline_smoothing(avg_timestamps, avg_data)
        #plt.plot(timestamps, data, 'o', filt_timestamps, filt_data, '-')
        #plt.show()
        
    return filt_data