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


"""
   SOM
"""
from minisom import MiniSom
#https://github.com/JustGlowing/minisom/blob/master/examples/PoemsAnalysis.ipynb

X_a=np.array(X)

map_dim = 100
som = MiniSom(map_dim, map_dim, 140, sigma=1.0, random_seed=1)
#som.random_weights_init(W)
som.train_batch(X_a, num_iteration=len(X)*500, verbose=True)


plt.figure(figsize=(20, 20))
# Plotting the response for each pattern in the iris dataset
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
#plt.colorbar()


t = np.zeros(len(Y_St), dtype=int)
t[Y_St == 'Base'] = 0
t[Y_St == 'Anes'] = 1
t[Y_St == 'Reco'] = 2

c = np.zeros(len(Y_ID), dtype=int)
c[Y_ID == 'WSAS17'] = 0
c[Y_ID== 'WSAS22'] = 0
c[Y_ID == 'WSAS13'] = 1
c[Y_ID == 'WSAS10'] = 0

# use different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2','C3','C4']
for cnt, xx in enumerate(X_a):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]], markerfacecolor='None',
             markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
plt.colorbar()
plt.axis([0, 50, 0, 50])

#plt.savefig('resulting_images/som_iris.png')
plt.show()

