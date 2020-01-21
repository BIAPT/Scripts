'''
    Code written by Parisa and Yacine Mahdid
'''

# General Import
import os
from os import listdir
import datetime
import time
import glob
import zipfile
import math
import shutil

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
#     "P14": "ABSENT",
#     "P15": "TP001354", #P15 collects garbage data
#     "session":"Session_Oct_31",
#     "num_participant":15,
#     "crashed_phones": [],
#     "phones_with_time_problem": ["orange"]
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
#     "P14":"ABSENT",
#     "P15":"ABSENT",
#     "session":"Session_Nov_7",
#     "num_participant":15,
#     "crashed_phones": ["orange"],
#     "phones_with_time_problem": ["orange"]    
# }


# #mappings for session Nov 14th
# experiment_info = {
#     "P1":"TP001376",
#     "P2":"TP001884",
#     "P3":"TP001353",
#     "P4":"TP001123",
#     "P5":"ABSENT",
#     "P6":"TP001689",
#     "P7":"TP001254",
#     "P8":"TP001822",
#     "P9":"ABSENT",
#     "P10":"TP001354",
#     "P11":"TP001472",
#     "P12":"TP001484",
#     "P13":"TP001491",
#     "P14":"ABSENT",
#     "P15":"ABSENT",
#     "session":"Session_Nov_14",
#     "num_participant":15,
#     "crashed_phones": ["yellow"],
#     "phones_with_time_problem": ["orange"]
# }

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
#     "P14":"TP001491",
#     "P15":"ABSENT",
#     "session":"Session_Nov_21",
#     "num_participant":15,
#     "crashed_phones": [],
#     "phones_with_time_problem": ["orange"]   
# }

# #mappings for session Dec 5th
# experiment_info = {
#     "P1":"TP001376",
#     "P2":"TP001884",
#     "P3":"TP001353",
#     "P4":"TP001491",
#     "P5":"TP001484",
#     "P6":"TP001689",
#     "P7":"TP001254",
#     "P8":"TP001822",
#     "P9":"ABSENT",
#     "P10":"TP001354",
#     "P11":"TP001472",
#     "P12":"ABSENT",
#     "P13":"ABSENT",
#     "P14":"TP001123",
#     "P15":"ABSENT",
#     "session":"Session_Dec_5",
#     "num_participant":15,
#     "crashed_phones": ["orange"],
#     "phones_with_time_problem": []    
# }

#mappings for session Dec 12th
experiment_info = {
    "P1":"TP001376",
    "P2":"TP001123",
    "P3":"TP001353",
    "P4":"ABSENT",
    "P5":"ABSENT",
    "P6":"TP001689",
    "P7":"TP001254",
    "P8":"TP001491",
    "P9":"ABSENT",
    "P10":"ABSENT",
    "P11":"TP001472",
    "P12":"TP001484",
    "P13":"ABSENT",
    "P14":"TP001822",
    "P15":"ABSENT",
    "session":"Session_Dec_12",
    "num_participant":15,
    "crashed_phones": ["orange"],
    "phones_with_time_problem": []    
}

# #mappings for session Dec 19th
# experiment_info = {
#     "P1":"TP001376",
#     "P2":"TP001884",
#     "P3":"TP001353",
#     "P4":"TP001354",
#     "P5":"ABSENT",
#     "P6":"TP001689",
#     "P7":"TP001254",
#     "P8":"TP001822",
#     "P9":"ABSENT",
#     "P10":"ABSENT",
#     "P11":"TP001472",
#     "P12":"TP001484",
#     "P13":"TP001123",
#     "P14":"TP001491",
#     "P15":"ABSENT",
#     "session":"Session_Dec_19",
#     "num_participant":15,
#     "crashed_phones": [],
#     "phones_with_time_problem": []    
# }




#change these based on your input and output paths
input_path = os.path.join("C:\\","Users","biomusic","Desktop", experiment_info["session"])
output_path = os.path.join("C:\\","Users","biomusic","Desktop","Nova", "data")
#output_path = os.path.join("C:\\","Users","biomusic","Desktop","Nova", "test output folder")


#take the name of crashed phones and phones with time problem and put them in local variables
crashed_phones = experiment_info["crashed_phones"]
phones_with_time_problem = experiment_info["phones_with_time_problem"]

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



already_glued = False
#if one of the phones crashed in the middle of the experiment
#and the experimenter has re-connected it and started recording again,
#this function is supposed to glue the data files from two recording folders together
def glue(crashed_phone, colorpath1):
    #set the path for the second part of the folder we want to glue
    colorpath2 = colorpath1.replace(crashed_phone + "1", crashed_phone + "2") + ".zip"

    #extract the second folder
    print("extracting..." + colorpath2)
    with zipfile.ZipFile(colorpath2, 'r') as zip_ref:
        colorpath2 = colorpath2.replace(".zip","") 
        zip_ref.extractall(colorpath2)

    #create a new folder, with the name of the creashed phone to put the glued data in
    outputcolor = os.path.join(input_path, crashed_phone)
    if not os.path.exists(outputcolor):
        os.makedirs(outputcolor)

    for sessionfolder1 in listdir(colorpath1):
        sessionpath1 = colorpath1 + os.sep + sessionfolder1
        #create the session folder in the creashed phone folder
        outputpath = os.path.join(input_path, crashed_phone, sessionfolder1)
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)

        for filename1 in listdir(sessionpath1):
            #we only need these three data. so we run through them to glueeeee
            if "_HR_" in filename1 or "_TEMP_" in filename1 or "_EDA_" in filename1:
                #the name of the output glued file can be the same as the files of the first part of the data
                outputfile = open(outputpath + os.sep + filename1, "w+")
                inputfile1 = open(sessionpath1 + os.sep + filename1, "r")
                #read each lines of the first part of data and write them on output file
                for line1 in inputfile1.readlines():
                    outputfile.write(line1)
                #keep the last line, to calculate the time gap between the files that is to be filled with NaN
                last_line = line1

                for sessionfolder2 in listdir(colorpath2):
                    sessionpath2 = colorpath2 + os.sep + sessionfolder2
                    for filename2 in listdir(sessionpath2):
                        #find the second part of the file that we are working on 
                        if filename1[-9:] ==  filename2[-9:]:
                            print(filename1)
                            print(filename2)
                            inputfile2 = open(sessionpath2 + os.sep + filename2, "r")
                            discard = inputfile2.readline() #discard the second file's header
                            first_line = inputfile2.readline()

                            #the last timestamp of the first file
                            timestamp1 = int(last_line.split(";")[0])
                            #the first timestamp of the second file
                            timestamp2 = int(first_line.split(";")[0])

                            
                            if "EDA" in filename1:
                                SamplingRate = 15
                            elif "TEMP" in filename1:
                                SamplingRate = 15
                            elif "HR" in filename1:
                                SamplingRate = 5
                            elif "BVP" in filename1:
                                SamplingRate = 300

                            #write NaN for the gap between two timestamps
                            i = 1
                            TS = timestamp1
                            while TS < timestamp2 - (1000/15):
                                TS = timestamp1 + round(i*1000*(1/SamplingRate))
                                if TS < timestamp2:
                                    outputfile.write(str(TS) + ";NaN" + '\n')
                                    i += 1

                            #continue writing the second file on the output file
                            outputfile.write(first_line)
                            for line2 in inputfile2.readlines():
                                outputfile.write(line2)

            #copy the session file that contains the information about TPS IDs to the output folder
            elif "session" in filename1:
                print("copying the session file...")
                shutil.copyfile(sessionpath1 + os.sep + filename1, outputpath + os.sep + filename1)
    
    return outputcolor


#Getting into the folder that contains the recording data.
#When importing data from each phone, after unzipping the compressed session file,
#it should be named as the color of the phone.
#So each color folder, contains another folder with the session name
#and inside that folder are the recorded data from that phone.
for colorfolder in listdir(input_path):
    if ".zip" in colorfolder:

        #sometimes, phone's system time might not be accurate. In that case, we have to move
        #the start recording time, based on the delay that phone had had
        for phone_with_time_problem in phones_with_time_problem:
            if phone_with_time_problem in colorfolder:
                start_time = load_time(os.path.join(input_path,"start_recording_time_" + phone_with_time_problem + ".txt"))
                phones_with_time_problem.remove(phone_with_time_problem)
            else:
                start_time = load_time(os.path.join(input_path,"start_recording_time.txt"))


        # Here we unzip the file
        print("extracting... " + colorfolder)
        color_path = input_path + os.sep + colorfolder
        with zipfile.ZipFile(color_path, 'r') as zip_ref:
            # we remove the .zip extension
            color_path = color_path.replace(".zip","") 
            #then we extract
            zip_ref.extractall(color_path)
            
        #if we have already glued using the first folder of this color,
        #skip this folder entirely, because its data is already glued and became stream files
        if already_glued:
            #make this false, for other crashed phones that might exist
            already_glued = False
            continue

        for crashed_phone in crashed_phones:
            #if this folder we are at, is either the first of second part of the crashed phone data
            if crashed_phone in colorfolder:
                if not already_glued:
                    print("glueing..." + crashed_phone)
                    #glue this file to its second folder, using "glue" function
                    #the function returns the path where the glued data is stored as the output
                    #so we continue creating stream files from the glued data.
                    color_path = glue(crashed_phone, color_path)
                    already_glued = True


        print(colorfolder)
        print("start time:")
        print(start_time)

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
