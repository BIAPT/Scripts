'''
    Code written by Yacine Mahdid using Parisa ruby code (tostream.rb)
    We are rewritting the ruby code because most of the pre-processing is already done in Python.
'''

# General Import
import os
import datetime
import time
import glob
import math

# Data manipulation imports
import csaps 
import numpy as np 

# Visualization imports
import matplotlib.pyplot as plt

# Input and ouput paths (set these up so that it works with your computer)
input_path = "C:/Users/biapt/Downloads/ruby_script_and_sample_data"
output_path = "C:/Users/biapt/Documents/GitHub/Scripts/SSI"

# Get the timestamp of when the script was run 
# We do this here so that all file created when this is run 
# will have the same timestamp
t_now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
t_utc = datetime.datetime.utcnow().strftime("%Y/%m/%d %H:%M:%S")


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
    current_sampling_rate = (x[-1] - x[0]) / (1000*len(x)) # The time is in milliseconds so we put them in seconds
    
    # get the ratio of augmentation
    ratio = new_sampling_rate / current_sampling_rate
    # Augment the data to match the wanted sampling rate
    filt_x = np.linspace(x[0], x[-1], ratio*len(x))

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
# TODO: This could be improved with slicing instead of accumulating an array
def window_average(timestamps, data, window_size=0.5):  
    '''
        window averaging with a jumping window (default is 0.5 seconds)
        the bigger the size of the window the smaller the new sample rate will be
        if we choose a window of 0.5 it will be 2Hz
    '''

    # convert the window size to be in milliseconds
    window_size = window_size*1000 

    # setting up the arrays and variables
    avg_timestamps = []
    avg_data = []
    current_i = 0

    # Iterating through each windows
    for start_t in range(start=timestamp[0], stop=timestamps[-1], step=window_size):
        
        # Adding the timestamp to be the middle of the average window
        avg_timestamps.append(start_t + (window_size/2))
        curr_data = []
        current_t = timestamp[current_i]
        end_t = start_t + window_size

        # Appending all the data within that window
        while current_t < end_t:
            curr_data.append(data[current_i])
            current_i = current_i + 1
            current_t = timestamp[current_i]

        avg_data.append(math.mean(curr_data))# TODO: Here this needs to be fixed (not internet right now)

    return (avg_timestamps, avg_data)



# Pre processing function that will apply the pre-processing technique
# based on what analysis technique we are working with.
def pre_process(timestamps, data, data_type, sample_rate):
    print("Preprocessing: " + data_type)

    # Window averaging at 2Hz (0.5seconds)
    (avg_timestamps, avg_data) = window_average(timestamps,data)

    # Pre-processing (if you want to tweak them change them here!)
    if data_type == "EDA":
        filt_data = one_euro(avg_timestamps, avg_data)
        plt.plot( timestamps, data, 'o', timestamps, filt_data, '-')
        plt.show()
    elif data_type == "TEMP":
        filt_data = exponential_decay(avg_data)
        plt.plot( timestamps, data, 'o', timestamps, filt_data, '-')
        plt.show()
    elif data_type == "HR":
        (filt_timestamps, filt_data) = cubic_spline_smoothing(avg_timestamps, avg_data)
        plt.plot(timestamps, data, 'o', filt_timestamps, filt_data, '-')
        plt.show()
        
    return filt_data


# Iterating through each TPS folder
directory_listing = glob.glob(input_path + os.sep + "TP00*")
for tps_path in directory_listing:
    # get the last foldername which is the name of the TPS
    tps_id = tps_path.split(os.sep)[-1] 
    print("TPS ID: " + str(tps_id))

    # Iterating through each files in the TPS folder
    file_listing = glob.glob(tps_path + os.sep + "2019-*.csv")
    for filename in file_listing:
        
        # skip the setting file
        if "setting" in filename:
            continue

        # Selecting right sample rate given the signal and data type
        if "BVP" in filename:
            data_type = "BVP"
            sample_rate = 300.0
        elif "EDA" in filename:
            data_type = "EDA"
            sample_rate = 15.0
        elif "TEMP" in filename:
            data_type = "TEMP"
            sample_rate = 15.0
        elif "HR" in filename:
            data_type = "HR"
            samplerate = 15.0
        elif "STR" in filename:
            data_type = "STR"
            samplerate = -1.0 # What do we do with this?
        else:
            raise Exception("Filename doesn't include BVP, EDA, TEMP, HR or STR! Please double check if this code is still valid for your raw signals.")

        # Creating the Stream~ file (containing the data)
        print("Input file: " + filename)
        stream_name = tps_id + "." + data_type + ".stream~"
        print("Output file: " + stream_name)

        # open our input and output file
        output_filename = output_path + os.sep + stream_name
        raw_file = open(filename, "r")


        # Read the input file line by line and
        num_row = 0 # keep track of the number of row written for the stream file
        raw_file.readline()
        
        raw_time = []
        raw_data = []
        for line in raw_file.readlines():
            # Strip nenwline and split the line
            line = line.replace(" ", "").rstrip()
            line = line.split(';')

            # What we need to do is check when the start time is what we have seted
            # above (however for now we didn't do it TODO!)
            if num_row == 0:
                start_time = int(line[0])

            # Get the timestamp and time point
            timestamp = int(line[0]) - start_time
            data = float(line[1])

            # Checking for nan value and replacing them with dummy value
            if(math.isnan(data)):
                data = 0 # Here we can set it to other thing than 0

            # Store the raw data
            raw_time.append(timestamp)
            raw_data.append(data)

            # Increase the number of row
            num_row+=1

        # Closing our file pointers
        raw_file.close()

        # Pre-process the data
        processed_data = pre_process(raw_time, raw_data, data_type, sample_rate)

        # write the data to the stream file
        stream_file = open(output_filename, "w")
        for data in processed_data:
            # Write each points on each line
            stream_file.write(str(data) + "\n")
        stream_file.close()

        # Creating the header stream file (no tilda ~)
        output_filename = stream_name.replace("stream~","stream")
        header_file = open(output_filename,"w")

        # Write the header
        tab = "\t" # for readability
        header_file.write("<?xml version=\"1.0\" ?>\n")
        header_file.write("<stream ssi-v=\"2\">\n")
        header_file.write(tab + "<info ftype=\"ASCII\" sr=\"" + str(sample_rate) + "\" dim=\"1\" byte=\"4\" type=\"FLOAT\" delim=\";\" />\n")
        header_file.write(tab + "<meta />\n") # Not sure why we need this XML tag?
        header_file.write(tab + "<time ms=\"0\" local=\"" + t_now + "\" system=\"" + t_utc + "\"/>\n")
        header_file.write(tab + "<chunk from=\"0.000000\" to=\"" + str(num_row/sample_rate) + "\" byte=\"0\" num=\"" + str(num_row) + "\"/>\n")
        header_file.write("</stream>\n")

        # Closing the file pointer
        header_file.close()

        
