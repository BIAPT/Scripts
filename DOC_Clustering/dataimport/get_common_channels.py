import scipy.io
import glob
import numpy as np
import pandas as pd

files = [f for f in glob.glob('data/WSAS_TIME_DATA_250Hz/Raw_250' + "**/*.mat", recursive=True)]

allchannels = np.arange(1, 129, 1)
channeldata = pd.DataFrame(files)

for c in allchannels:
    channeldata['E' + str(c)] = np.zeros(len(files))

channeldata['Cz'] = np.zeros(len(files))

n = 0
for i in files:
    if (i.find("WSAS02")) == -1:
        recording = scipy.io.loadmat(i)
        recording = recording['EEG']
        chloc = recording['chanlocs'][0][0][0]
        chloc = pd.DataFrame(chloc)
        channels = []
        for c in range(0, len(chloc)):
            chan = chloc.iloc[c, 0][0]
            channels.append(chan)
        for c in channels:
            channeldata[c][n] = 1
    n = n + 1


np.savetxt('data/WSAS_TIME_DATA_250Hz/channeldata.txt', channeldata.values, fmt='%s')
occurence = np.sum(channeldata.iloc[:, 1:], axis=0)
np.savetxt('data/WSAS_TIME_DATA_250Hz/channeldata_sum.txt', occurence.values, fmt='%s')
