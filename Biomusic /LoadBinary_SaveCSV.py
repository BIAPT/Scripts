#!/usr/bin/env python
# coding: utf-8

# #### Dannie Fu September 22 2020
# #### This file converts binary file into csv file and saves the csv

# In[ ]:


import pandas as pd 
import struct,os,sys
import glob
import csv 

def getTime(startTime, offset, fs):
    return startTime + offset*1000/fs

# Sampling freqs
fs_ACC = 75
fs_BVP = 300
fs_EDA = 15
fs_HRV = 4
fs_STR = 15
fs_TEMP = 15

# Get list of files in session directory 
sessionDir = '/Volumes/Seagate/Tuning In /MWTI002/2019-12-13/part3/'
os.chdir(sessionDir)

# Get start time 
sessionFile = sessionDir + 'sessions.csv'
startTime = pd.read_csv(sessionFile, sep=";",usecols=["startTime"])

for file in glob.glob("*[!.csv]"):
    print(file)

    prefix = file.split("._")[0]
    suffix = file.split("._")[1]
    f = open(file,"rb")
    myStruct=struct.Struct(suffix)

    myList = []
    time = []
    
    # Append header names
    if (file.find('EDA') != -1):
        time.append('eda_time')
        myList.append('eda_data')
    elif (file.find('ACC') != -1):
        time.append('acc_time')
        myList.append(['acc_x', 'acc_y','acc_z'])
    elif (file.find('BVP') != -1):
        time.append('bvp_time')
        myList.append('bvp_data')
    elif (file.find('HRV') != -1):
        time.append('hrv_time')
        myList.append(['hrv_x','hrv_y','hrv_z'])
    elif (file.find('STR') != -1):
        time.append('str_time')
        myList.append('str_data')
    elif (file.find('TEMP') != -1 and file.find('TEMPR') == -1):
        time.append('temp_time')
        myList.append('temp_data')    
    elif (file.find('HR') != -1 and file.find('HRV') == -1):
        myList.append(['hr_time', 'hr_data'])  
    elif (file.find('EDR') != -1):
        myList.append(['edr_peak_time', 'edr_peak_dat','edr_rise_time','edr_max_deriv','edr_amplitude','edr_max_deriv2'])  
    elif (file.find('TEMPR') != -1):
        myList.append(['tempr_peak_time','tempr_peak_dat','tempr_rise_time','tempr_max_deriv','tempr_amplitude','tempr_max_deriv2'])
    
    i=0
    # Append times and data
    while True:
        data=f.read(myStruct.size)
        
        if len(data)==0: break
            
        a=[*myStruct.unpack(data)]
        
        if (file.find('EDA') != -1):
            time.append(getTime(startTime.startTime[0], i, fs_EDA))
            myList.append(a[0])
        elif (file.find('ACC') != -1):
            time.append(getTime(startTime.startTime[0], i, fs_ACC))
            myList.append(a)
        elif (file.find('BVP') != -1):
            time.append(getTime(startTime.startTime[0], i, fs_BVP))
            myList.append(a[0])
        elif (file.find('HRV') != -1):
            time.append(getTime(startTime.startTime[0], i, fs_HRV))
            myList.append(a)
        elif (file.find('STR') != -1):
            time.append(getTime(startTime.startTime[0], i, fs_STR))
            myList.append(a[0])
        elif (file.find('TEMP') != -1 and file.find('TEMPR') == -1):
            time.append(getTime(startTime.startTime[0], i, fs_TEMP))
            myList.append(a[0])
        elif (file.find('HR') != -1 and file.find('HRV') == -1):
            myList.append(a)  
        elif (file.find('EDR') != -1):
            myList.append(a)  
        elif (file.find('TEMPR') != -1):
            myList.append(a)  
        
        i+=1
        
    if time:
        df1 = pd.DataFrame(time)
        df2 = pd.DataFrame(myList)
        df = pd.concat([df1, df2],axis=1)
    else:
        df = pd.DataFrame(myList)
        
    new_header = df.iloc[0] #grab the first row for the header
    df = df[1:] #take the data less the header row
    df.columns = new_header #set the header row as the df header
    
    df.to_csv(prefix +".csv", index=False) 
    


# In[ ]:




