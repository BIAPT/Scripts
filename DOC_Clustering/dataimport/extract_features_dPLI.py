import pandas as pd
import numpy as np


def extract_single_features(X_step,channels,selection_1,selection_2,name):
    if len(X_step.shape) == 3:
        selected_1=[]
        selected_2=[]
        for i in range(0,len(selection_1)):
            try:
                selected_1.append(np.where(channels==selection_1[i])[0][0])
            except:
                print("An exception occurred: Electrode" + str(selection_1[i]) +' does not exist in  ' +name)

        for i in range(0,len(selection_2)):
            try:
                selected_2.append(np.where(channels==selection_2[i])[0][0])
            except:
                print("An exception occurred: Electrode" + str(selection_2[i]) +' does not exist in  ' +name)

        dPLI=[]

        for a in selected_1:
            for b in selected_2:
                if a!= b:
                    dPLI.append(X_step[:,min(a,b),max(a,b)])


        return np.mean(dPLI)

    if len(X_step.shape) == 2:
        selected_1 = []
        selected_2 = []
        for i in range(0, len(selection_1)):
            try:
                selected_1.append(np.where(channels == selection_1[i])[0][0])
            except:
                print("An exception occurred: Electrode" + str(selection_1[i]) +' does not exist in  ' +name)

        for i in range(0, len(selection_2)):
            try:
                selected_2.append(np.where(channels == selection_2[i])[0][0])
            except:
                print("An exception occurred: Electrode" + str(selection_2[i]) +' does not exist in  ' +name)

        dPLI = []

        for a in selected_1:
            for b in selected_2:
                if a != b:
                    dPLI.append(X_step[min(a, b), max(a, b)])

        return np.mean(dPLI)


def get_difference (data):
    tofill= np.zeros((data.shape[0]-1, data.shape[1]))
    for i in range(0,data.shape[0]-1):
        j=i+1
        tofill[i,:]=data[j]-data[i]
    return  tofill



