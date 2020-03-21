import scipy.io
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import pandas as pd
import stability_measure
from matplotlib import pyplot as plt
import random
import pickle

data=pd.read_pickle('data/final_wPLI_clustering.pickle')
X=data.iloc[:,4:]
Y_ID=data.iloc[:,1]
Y_St=data.iloc[:,2]
Y_time=data.iloc[:,3]

data_reco=data[(Y_ID == 'WSAS19')|(Y_ID == 'WSAS20')|(Y_ID == 'WSAS02')|(Y_ID == 'WSAS09')]
X_reco=data_reco.iloc[:,4:]
Y_ID_reco=data_reco.iloc[:,1]

data_acute=data[(Y_ID == 'WSAS05') | (Y_ID == 'WSAS10') | (Y_ID == 'WSAS18') | (Y_ID == 'WSAS12')]
X_acute=data_acute.iloc[:,4:]
Y_ID_acute=data_acute.iloc[:,1]



"""
Stability Index
"""
P=[3,4,5,6,7,8,9,10]     #number of Principal components to iterate
K=[2,3,4,5,6,7,8,9,10]     #number of K-clusters to iterate
Rep=20         #number of Repetitions (Mean at the end)

#[SI_M ,SI_SD] = stability_measure.compute_stability_index(X,Y_ID, P, K, Rep)
[SI_M_reco ,SI_SD_reco] = stability_measure.compute_stability_index(X_reco,Y_ID_reco, P, K, Rep)
[SI_M_acute ,SI_SD_acute] = stability_measure.compute_stability_index(X_acute,Y_ID_acute, P, K, Rep)


#pd.DataFrame(SI_M).to_pickle('data/SI_M_100x100x3-10x2-10.pickle')
#pd.DataFrame(SI_SD).to_pickle('data/SI_SD_100x100x3-10x2-10.pickle')


fig,a =  plt.subplots(2,2)
plt.setp(a, xticks=[0,1,2,3,4,5,6,7,8,9] , xticklabels=['2','3','4','5','6','7','8','9','10'],
        yticks=[0,1,2,3,4,5,6,7,8], yticklabels= ['3','4','5','6','7','8','9','10'],
         xlabel= 'K-Clusters',ylabel='Principle Components')
im=a[0][0].imshow(np.transpose(SI_M_reco))
a[0][0].set_title('Stability Index Mean: Recovered patients')
plt.colorbar(im,ax=a[0,0])
im=a[0][1].imshow(np.transpose(SI_SD_reco))
a[0][1].set_title('Stability Index SD: Recovered patients')
plt.colorbar(im,ax=a[0,1])
im=a[1][0].imshow(np.transpose(SI_M_acute))
a[1][0].set_title('Stability Index Mean: Acute patients')
plt.colorbar(im,ax=a[1,0])
im=a[1][1].imshow(np.transpose(SI_SD_acute))
a[1][1].set_title('Stability Index SD: Acute patients')
plt.colorbar(im,ax=a[1,1])


