import time

timefile = open("startrecordingtime.txt", "w+")
timefile.write(str(time.time()))