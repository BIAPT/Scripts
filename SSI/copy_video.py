import os
from os import listdir
from shutil import copyfile


#make sure to put the video file in the session folder before running this code

session =  "Session_Dec_19"
input_Path = os.path.join("C:\\","Users","biomusic","Desktop", session)
output_path = os.path.join("C:\\","Users","biomusic","Desktop","Nova", "data")

#the name of the video file must be "user.video.MPG". Rename it to this if it has another name for some reason.
for inputfile in listdir(input_Path):
	if "MPG" in inputfile and "user.video.MPG" not in inputfile:
		os.rename(os.path.join(input_Path,inputfile), os.path.join(input_Path,"user.video.MPG"))
		print("renamed the video file to \"user.video.MPG\"")

#navigate in participants folder
for Pfolder in listdir(output_path):
	P_path = os.path.join(output_path,Pfolder)
	for Sfolder in listdir(P_path):
		#if you find this session's folder
		if Sfolder == session:
			S_path = os.path.join(P_path,Sfolder)
			#copy the video on it
			print("Copying video of " + session + " to the folder " + Pfolder)
			copyfile(os.path.join(input_Path,"user.video.MPG"), os.path.join(S_path,"user.video.MPG"))
			print("Done!")







