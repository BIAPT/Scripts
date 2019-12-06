'''
    Code written by Parisa and Yacine Mahdid ruby code (tostream.rb)
    We are rewritting the ruby code because most of the pre-processing is already done in Python.
'''

# General Import
import os
from os import listdir
import datetime
import time
import glob
import zipfile
import math

# Data management import
import pandas as pd

# Filters Imports
import sys

# Utils import
# This is where most of the helper methods are, I did that to increase the readability of the code
from utils import load_time, write_header, Experiment

# Signal analysis import
from signals import pre_process


# #mappings for session Oct 31st
# experiment_info = {
#     "P1":"TP001689",
#     "P2":"TP001491",
#     "P3":"TP001353",
#     "P4":"TP001123",
#     "P5":"TP001484",
#     "P6":"TP001376",
#     "P7":"TP001254",
#     "P8":"TP001822",
#     "P9":"ABSENT",
#     "P10":"ABSENT",
#     "P11":"ABSENT",
#     "P12":"TP001472",
#     "P13":"ABSENT",
#     "P14": "TP001354", #gargabe data from this sensor
#     "session":"Session_Oct_31",
#     "num_participant":14
# }


# #mappings for session Nov 7th
# experiment_info = {
#     "P1":"TP001689",
#     "P2":"ABSENT",
#     "P3":"TP001353",
#     "P4":"ABSENT",
#     "P5":"ABSENT",
#     "P6":"TP001376",
#     "P7":"TP001254",
#     "P8":"TP001822",
#     "P9":"TP001491",
#     "P10":"TP001354",
#     "P11":"TP001472",
#     "P12":"TP001884",
#     "P13":"TP001123",
#     "session":"Session_Nov_7",
#     "num_participant":13
# }


#mappings for session Nov 14th
experiment_info = {
    "P1":"TP001376",
    "P2":"TP001884",
    "P3":"TP001353",
    "P4":"TP001123",
    "P5":"ABSENT",
    "P6":"TP001689",
    "P7":"TP001254",
    "P8":"TP001822",
    "P9":"ABSENT",
    "P10":"TP001354",
    "P11":"TP001472",
    "P12":"TP001484",
    "P13":"TP001491",
    "session":"Session_Nov_14",
    "num_participant":13
}

# #mappings for session Nov 21st
# experiment_info = {
#     "P1":"TP001376",
#     "P2":"TP001884",
#     "P3":"TP001353",
#     "P4":"ABSENT",
#     "P5":"ABSENT",
#     "P6":"TP001689",
#     "P7":"TP001254",
#     "P8":"TP001822",
#     "P9":"ABSENT",
#     "P10":"TP001354",
#     "P11":"TP001472",
#     "P12":"TP001484",
#     "P13":"TP001123",
#     "P14":"ABSENT",
#     "P15":"TP001491",
#     "session":"Session_Nov_21",
#     "num_participant":15
# }

# #mappings for session Dec 5th
# experiment_info = {
#     "P1":"TP001376",
#     "P2":"TP001884",
#     "P3":"TP001353",
#     "P4":"ABSENT",
#     "P5":"ABSENT",
#     "P6":"TP001689",
#     "P7":"TP001254",
#     "P8":"TP001822",
#     "P9":"TP001491",
#     "P10":"TP001354",
#     "P11":"TP001472",
#     "P12":"TP001484",
#     "P13":"TP001123",
#     "P14":"ABSENT",
#     "P15":"ABSENT",
#     "session":"Session_Dec_5",
#     "num_participant":15
# }




#Input and ouput paths (set these up so that it works with your computer)
input_path = os.path.join("C:\\","Users","biomusic","Desktop", experiment_info["session"])
output_path = os.path.join("C:\\","Users","biomusic","Desktop","Nova", "data")
# input_path = os.path.join("C:\\","Users","biomusic","Desktop", "test_session")
# output_path = os.path.join("C:\\","Users","biomusic","Desktop","Nova", "only to test the python code")


# input_path = os.path.join("C:\\","Users","biapt","Documents","GitHub","Scripts","SSI","test_data",experiment_info["session"])
# output_path = os.path.join("C:\\","Users","biapt","Documents","GitHub","Scripts","SSI","test_out")

# Get the timestamp of when the script was run 
# We do this here so that all file created when this is run 
# will have the same timestamp
t_now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
t_utc = datetime.datetime.utcnow().strftime("%Y/%m/%d %H:%M:%S")

# In unix time, this is used to ignore some data
start_time = load_time(os.path.join(input_path,"start_recording_time.txt"))

# We setup the experiment using the experiment_info defined above
# this will take care of structuring our file structure
experiment = Experiment(experiment_info, output_path)


#Getting into the folder that contains the recording data.
#When importing data from each phone, after unzipping the compressed session file,
#it should be named as the color of the phone.
#So each color folder, contains another folder with the session name
#and inside that folder are the recorded data from that phone.
for colorfolder in listdir(input_path):
    if ".zip" in colorfolder:
        if "orange" in colorfolder:
            start_time = load_time(os.path.join(input_path,"start_recording_time_orange.txt"))
        else:
            start_time = load_time(os.path.join(input_path,"start_recording_time.txt"))
        print(colorfolder)
        print("start time:")
        print(start_time)


        # Here we unzip the file
        print("extracting... " + colorfolder)
        color_path = input_path + os.sep + colorfolder
        with zipfile.ZipFile(color_path, 'r') as zip_ref:
            # we remove the .zip extension
            color_path = color_path.replace(".zip","") 
            #then we extract
            zip_ref.extractall(color_path)
        
        
        # Here we iterate through the color path session folder
        for sessionfolder in listdir(color_path):
            session_path = color_path + os.sep + sessionfolder
            sessions_path = os.path.join(session_path,"sessions.csv")
            sessionfile = open(sessions_path, "r")
            
            # Here we are using pandas to load the data into an easy to manage
            # dataframe
            session_df = pd.read_csv(sessionfile, delimiter=";")
            tps_names = session_df['deviceNames'][0].split(',')
            TPS_to_one_phone = len(tps_names)

            # Get all the participants we need for the analysis
            # set their position too using tps_names
            curr_participants = []
            for i in range(0,len(tps_names)):
                print(tps_names[i])
                # get the participant by tps name
                participant = experiment.get_participant(tps_names[i])
                # set his position in the experiment
                participant.set_position(i+1)
                # add to current participants
                curr_participants.append(participant)

            # Iterate through each recording files
            for filename in listdir(session_path):
                
                # Get the participant that match the current file
                if filename.endswith("1.csv"):
                    curr_participant = curr_participants[0]
                elif filename.endswith("2.csv"):
                    curr_participant = curr_participants[1]
                elif filename.endswith("3.csv"):
                    curr_participant = curr_participants[2]
                elif filename.endswith("4.csv"):
                    curr_participant = curr_participants[3]
                else:
                    # We skip it if it doesn't contain data
                    continue
                    
                participant_dir = curr_participant.saving_path
                if not os.path.exists(participant_dir):
                    os.makedirs(participant_dir)
                if "HRV" in filename or "BVP" in filename or "TEMPR" in filename or "STR" in filename: 
                    continue # skip hrv too
                elif "EDA" in filename:
                    data_type = "EDA"
                    sample_rate = 2.0
                elif "TEMP" in filename:
                    data_type = "TEMP"
                    sample_rate = 2.0
                elif "HR" in filename:
                    data_type = "HR"
                    samplerate = 2.0
                else:
                    continue

                # open our input file
                raw_file = open(session_path + os.sep + filename, "r")
                #open .stream and .stream~ output files with the naming that NOVA likes (role.type.stream)
                stream_name = os.path.join(participant_dir, "user." + data_type + ".stream~")
                stream_file = open(stream_name, "w")
                print("Output file: " + stream_name)
                print("Output file: " + stream_name.replace("stream~", "stream"))


                # Read the input file line by line and
                num_row = 0 # keep track of the number of row written for the stream file
                raw_file.readline()
                
                raw_time = []
                raw_data = []
                for line in raw_file.readlines():
                    # Strip nenwline and split the line
                    line = line.replace(" ", "").rstrip()
                    line = line.split(';')

                    # Get the timestamp and time point
                    
                    timestamp = int(line[0]) - start_time 
                    data = float(line[1])
                    # If this timestamp is negative meaning its before the time we care about
                    # we skip it
                    if timestamp < 0:
                        continue

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
                processed_data = pre_process(raw_time, raw_data, data_type)

                # write the data to the stream file
                for data_pts in processed_data:
                    # Write each points on each line
                    stream_file.write(str(data_pts) + "\n")
                stream_file.close()

                # Write the header
                # For information about the header see utils.py
                num_row = len(processed_data)
                new_sample_rate = 2 # in Hz
                header_file = open(stream_name.replace("stream~","stream"),"w")
                write_header(header_file, new_sample_rate, t_now, t_utc, num_row)
                header_file.close()
