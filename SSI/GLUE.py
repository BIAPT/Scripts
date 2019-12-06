import os
from os import listdir

for filename1 in listdir("one"):
	outputfile = open("final/" + filename1, "w")
	inputfile1 = open("one/" + filename1, "r")
	num_row = 0
	line1 = ""
	for line1 in inputfile1.readlines():
		num_row+=1
		outputfile.write(line1)
	last_line = line1

	for filename2 in listdir("two"):
		if filename1[-9:] ==  filename2[-9:]:
			print(filename1)
			print(filename2)
			inputfile2 = open("two/" + filename2, "r")
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
					num_row+=1

			#continue writing the second file
			outputfile.write(first_line)
			for line2 in inputfile2.readlines():
				num_row+=1
				outputfile.write(line2)

