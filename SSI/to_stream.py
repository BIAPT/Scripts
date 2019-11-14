'''
    Code written by Yacine Mahdid using Parisa ruby code (tostream.rb)
    We are rewritting the ruby code because most of the pre-processing is already done in Python.
'''

# General Import
import os
from os import listdir
import datetime
import time
import glob
import zipfile


# Filters Imports
import sys

# Utils import
# This is where most of the helper methods are, I did that to increase the readability of the code
# If you need to edit these functionality go to ./utils.py
# - load_time
# - write_header
from utils import load_time, write_header

# Signal analysis import
from signals import pre_process

# #settnings for session Oct 31st
# TPSDictionary = [
# ["TPS001689" , "P1"],
# ["TPS001491" , "P2"],
# ["TPS001353" , "P3"],
# ["TPS001123" , "P4"],
# ["TPS001484" , "P5"],
# ["TPS001376" , "P6"],
# ["TPS001254" , "P7"],
# ["TPS001822" , "P8"],
# ["ABSENT" , "P9"],
# ["ABSENT" , "P10"],
# ["ABSENT" , "P11"],
# ["TPS001472" , "P12"],
# ["ABSENT" , "P13"]]
# session = "Session_Oct_31"


#mappings for session Nov 7th
TPSDictionary = [
["TPS001689" , "P1"],
["ABSENT" , "P2"],
["TPS001353" , "P3"],
["ABSENT" , "P4"],
["ABSENT" , "P5"],
["TPS001376" , "P6"],
["TPS001254" , "P7"],
["TPS001822" , "P8"],
["TPS001491" , "P9"],
["TPS001354" , "P10"],
["TPS001472" , "P11"],
["TPS001884" , "P12"],
["TPS001123" , "P13"]]
session = "Session_Nov_7"



# Input and ouput paths (set these up so that it works with your computer)
# input_path = os.path.join("C:\\","Users","biomusic","Desktop", session)
# # output_path = os.path.join("C:\\","Users","biomusic","Desktop","Nova", "data")
# output_path = os.path.join("C:\\","Users","biomusic","Desktop","Nova", "only to test the python code")

input_path = os.path.join("C:\\","Users","biapt","Documents","GitHub","Scripts","SSI","test_data",session)
output_path = os.path.join("C:\\","Users","biapt","Documents","GitHub","Scripts","SSI","test_out")

TPS1 = ""
TPS2 = ""
TPS3 = ""
TPS4 = ""
TPS1Dir = ""
TPS2Dir = ""
TPS3Dir = ""
TPS4Dir = ""


# Get the timestamp of when the script was run 
# We do this here so that all file created when this is run 
# will have the same timestamp
t_now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
t_utc = datetime.datetime.utcnow().strftime("%Y/%m/%d %H:%M:%S")

# In unix time, this is used to ignore some data
start_time = load_time("start_recording_time.txt")


#Getting into the folder that contains the recording data.
#When importing data from each phone, after unzipping the compressed session file,
#it should be named as the color of the phone.
#So each color folder, contains another folder with the session name
#and inside that folder are the recorded data from that phone.
for colorfolder in listdir(input_path):
    # If we see a file that doesn't end with zip we skip it
    if not ".zip" in colorfolder:
        continue

    # Here we unzip the file
    color_path = input_path + os.sep + colorfolder
    with zipfile.ZipFile(color_path, 'r') as zip_ref:
        print("extracting... " + colorfolder)
        zip_ref.extractall(input_path)
        color_path = color_path.replace(".zip","") # we remoze the .zip extension
    
    # Here we iterate through the color path session folder
    for sessionfolder in listdir(color_path):
        session_path = color_path + os.sep + sessionfolder
        for root, dirs, recordingfiles in os.walk(session_path):
            for filename in recordingfiles:
                #calculate the number of sensors that are connected to that phone by reading the end of filename
                #it can be 2 or 3 or 4 sensors, but you can always add more (not recommended)
                if filename.endswith("4.csv"):
                    TPS_to_one_phone = 4
                elif filename.endswith("3.csv"):
                    TPS_to_one_phone = 3
                elif filename.endswith("2.csv"):
                    TPS_to_one_phone = 2

                #the recorded files are not named based on the TPS we got recording from
                #however, the name of the TPSs are stored in session file
                #so we have to read inside of it and create the folders with the corresponding TPS name
                if "session" in filename:
                    sessionfile = open(session_path + os.sep + filename, "r")
                    contents =sessionfile.read()
                    if TPS_to_one_phone == 2:
                        TPS1 = "TPS" + contents.split("TP")[1].split(",")[0]
                        TPS2 = "TPS" + contents.split("TP")[2].split(";")[0]
                        tpss = [TPS1,TPS2,"",""]

                    elif TPS_to_one_phone == 3:
                        TPS1 = "TPS" + contents.split("TP")[1].split(",")[0]
                        TPS2 = "TPS" + contents.split("TP")[2].split(",")[0]
                        TPS3 = "TPS" + contents.split("TP")[3].split(";")[0]
                        tpss = [TPS1,TPS2,TPS3,""]

                    elif TPS_to_one_phone == 4:
                        TPS1 = "TPS" + contents.split("TP")[1].split(",")[0]
                        TPS2 = "TPS" + contents.split("TP")[2].split(",")[0]
                        TPS3 = "TPS" + contents.split("TP")[3].split(",")[0]
                        TPS4 = "TPS" + contents.split("TP")[4].split(";")[0]
                        tpss = [TPS1,TPS2,TPS3,TPS4]


                    for [TPSID,ParID] in TPSDictionary:
                        if TPSID == TPS1:
                            TPS1Dir = os.path.join(output_path,ParID,session)
                            if not os.path.exists(TPS1Dir):
                                os.mkdir(TPS1Dir)
                            # print("Sensor: " + TPS1 + " on the phone color: " + colorfolder + " for participant: " + ParID)
                        if TPSID == TPS2:
                            TPS2Dir = os.path.join(output_path,ParID,session)
                            if not os.path.exists(TPS2Dir):
                                os.mkdir(TPS2Dir)
                            # print("Sensor: " + TPS2 + " on the phone color: " + colorfolder + " for participant: " + ParID)
                        if tpss[2] != "":
                            if TPSID == TPS3:
                                TPS3Dir = os.path.join(output_path,ParID,session)
                                if not os.path.exists(TPS3Dir):
                                    os.mkdir(TPS3Dir)
                                # print("Sensor: " + TPS1 + " on the phone color: " + colorfolder + " for participant: " + ParID)
                            if tpss[3] != "":
                                if TPSID == TPS4:
                                    TPS4Dir = os.path.join(output_path,ParID,session)
                                    if not os.path.exists(TPS4Dir):
                                        os.mkdir(TPS4Dir)
                                # print("Sensor: " + TPS2 + " on the phone color: " + colorfolder + " for participant: " + ParID)


            for filename in recordingfiles:
                #the recorded files end either with 1 or 2 which shows the file was recorded
                #either from the first TPS on the session file or the second TPS
                if filename.endswith("1.csv"):
                    participant_dir = TPS1Dir
                elif filename.endswith("2.csv"):
                    participant_dir = TPS2Dir
                elif filename.endswith("3.csv"):
                    participant_dir = TPS3Dir
                elif filename.endswith("4.csv"):
                    participant_dir = TPS4Dir
                    

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
                    continue

                # open our input file
                raw_file = open(session_path + os.sep + filename, "r")

                #open .stream and .stream~ output files with the naming that NOVA likes (role.type.stream)
                stream_name = participant_dir + os.sep + "user." + data_type + ".stream~"
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

                    # get the timestamp and the data points
                    raw_time.append(int(line[0]))
                    raw_data.append(float(line[1]))

                    # Increase the number of row
                    num_row+=1

                # Closing our file pointers
                raw_file.close()

                # Pre-process the data
                processed_data = pre_process(raw_time, raw_data, data_type, sample_rate)

                # write the data to the stream file
                for data_pts in processed_data:
                    # Write each points on each line
                    stream_file.write(str(data_pts) + "\n")
                stream_file.close()

                # Write the header
                # For information about the header see utils.py
                header_file = open(stream_name.replace("stream~","stream"),"w")
                write_header(header_file, sample_rate, t_now, t_utc, num_row)
                header_file.close()
