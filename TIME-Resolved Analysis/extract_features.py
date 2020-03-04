import pandas as pd
import numpy as np

def extract_features (data,getmean=False):
    data=data

    # Calculate nr of features with gaussian sum formula
    # because we don't take the diagonal as a feature
    nr_electrodes = data.shape[1]
    nr_features = int(((nr_electrodes - 1) ** 2 + (nr_electrodes - 1)) / 2)

    # create empty dataframe for features
    tofill = np.zeros((data.shape[0], nr_features))

    if len(data.shape)==3:
        timesteps=data.shape[0]

        # fill rows with diagonals features
        for t in range(0, timesteps):
            tmp = []
            for e in range(1, nr_electrodes):
                tmp.extend(data[t].diagonal(e))
            tofill[t, :] = tmp
        if getmean == True:
            tofill = np.mean(tofill, axis=1)

    if len(data.shape) == 2:
        timesteps = 1

        # fill rows with diagonals features
        tmp = []
        for e in range(1, nr_electrodes):
            tmp.extend(data.diagonal(e))
        tofill = tmp

        if getmean == True:
            tofill = np.mean(tofill, axis=0)

    return tofill

def extract_single_features(X_step,channels,selection):
    if len(X_step.shape) == 3:
        selected=[]
        for i in range(0,len(selection)):
            selected.append(np.where(channels==selection[i])[0][0])
        X_step=X_step[:,selected,:] #[time,horizontal,vertical]
        X_step=X_step[:,:,selected]
        return X_step

    if len(X_step.shape) == 2:
        selected=[]
        for i in range(0,len(selection)):
            selected.append(np.where(channels==selection[i])[0][0])
        X_step=X_step[selected,:] #[time,horizontal,vertical]
        X_step=X_step[:,selected]
        return X_step


def get_difference (data):
    tofill= np.zeros((data.shape[0]-1, data.shape[1]))
    for i in range(0,data.shape[0]-1):
        j=i+1
        tofill[i,:]=data[j]-data[i]
    return  tofill



