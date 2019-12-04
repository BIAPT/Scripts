import time
import os

session =  "Session_Dec_5"
this_session_folder_path = os.path.join("C:\\","Users","biomusic","Desktop", session)
os.mkdir(this_session_folder_path)

timefile = open(os.path.join(this_session_folder_path, "start_recording_time.txt"), "w+")
timefile.write(str(time.time()*1000))