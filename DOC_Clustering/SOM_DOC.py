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


plt.figure(figsize=(15, 20))
# Plotting the response for each pattern in the iris dataset
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
#plt.colorbar()


t = np.zeros(len(Y_St), dtype=int)
t[Y_St == 'Base'] = 0
t[Y_St == 'Anes'] = 1
t[Y_St == 'Reco'] = 2

c = np.zeros(len(Y_ID), dtype=int)
c[Y_ID == 'WSAS10'] = 1
c[Y_ID== 'WSAS11'] = 2
c[Y_ID == 'WSAS18'] = 3
c[Y_ID == 'WSAS12'] = 4
c[Y_ID == 'WSAS05'] = 5

# use different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2','C3','C4','C5']
for cnt, xx in enumerate(X_a):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]], markerfacecolor='None',
             markeredgecolor=colors[c[cnt]], markersize=12, markeredgewidth=2)
plt.colorbar()
plt.axis([0, 100, 0, 100])

#plt.savefig('resulting_images/som_iris.png')
plt.show()


from mpl_toolkits.mplot3d import Axes3D


# create a 10 x 10 vertex mesh
xx, yy = np.meshgrid(np.linspace(0,99,100), np.linspace(0,99,100))

# create vertices for a rotated mesh (3D rotation matrix)
X1 =  xx
Y =  yy
Z =  som.distance_map().T

# create some dummy data (20 x 20) for the image
data = som.distance_map().T

# create the figure
fig = plt.figure()

# show the reference image
ax1 = fig.add_subplot(121)
ax1.imshow(data, cmap=plt.cm.BrBG, interpolation='nearest', origin='lower')

# show the 3D rotated projection
ax2 = fig.add_subplot(122, projection='3d')
cset = ax2.contourf(X1, Y, Z,500)
for cnt, xx in enumerate(X_a):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    ax2.scatter(w[0], w[1],Z[w[1], w[0]],color=colors[t[cnt]])

plt.colorbar(cset)
plt.show()