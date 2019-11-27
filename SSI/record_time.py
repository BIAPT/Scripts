import time

timefile = open("start_recording_time.txt", "w+")
timefile.write(str(time.time()*1000))