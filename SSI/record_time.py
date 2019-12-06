import time
import os

#this code will store the start recording time in the session folder (where all data is)

session =  "Session_Dec_5"
this_session_folder_path = os.path.join("C:\\","Users","biomusic","Desktop", session)
if not os.path.exists(this_session_folder_path):
	os.mkdir(this_session_folder_path)

timefile = open(os.path.join(this_session_folder_path, "start_recording_time.txt"), "w+")
timefile.write(str(time.time()*1000))