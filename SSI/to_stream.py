'''
    Code written by Yacine Mahdid using Parisa ruby code (tostream.rb)
    We are rewritting the ruby code because most of the pre-processing is already done in Python.
'''

# General Import
import os
import datetime
import time
import glob

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
def cubic_spline_smoothing(x, y, p=0.01, new_sampling_rate=300):
    '''
        We are using the csaps python package which is a port of the matlab
        using this port https://pypi.org/project/csaps/
        
        p is the smoothing parameter, we can decrease it for more smoothing
        however 0.001 is what we used previously
        
        new_sampling_rate is how much more data points (in Hz) we want to interpolate the gaps

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

# Pre processing function that will apply the pre-processing technique
# based on what analysis technique we are working with.
def pre_process(timestamps, data_pts, data_type, sample_rate):
    print("Preprocessing: " + data_type)

    # For all of them we will average with a window size of 0.5seconds
    # TODO: do window average

    # Pre-processing (if you want to tweak them change them here!)
    if data_type == "EDA":
        data_pts = data_pts
    elif data_type == "TEMP":
        data_pts = data_pts
    elif data_type == "HR":

        (filt_timestamps, filt_data) = cubic_spline_smoothing(timestamps, data_pts)
        plt.plot(timestamps, data_pts, 'o', filt_timestamps, filt_data, '-')
        plt.show()
        
    return data_pts


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

            # get the timestamp and the data points
            raw_time.append(int(line[0]) - start_time)
            raw_data.append(float(line[1]))

            # Increase the number of row
            num_row+=1

        # Closing our file pointers
        raw_file.close()

        # Pre-process the data
        processed_data = pre_process(raw_time, raw_data, data_type, sample_rate)

        # write the data to the stream file
        stream_file = open(output_filename, "w")
        for data_pts in processed_data:
            # Write each points on each line
            stream_file.write(str(data_pts) + "\n")
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

        