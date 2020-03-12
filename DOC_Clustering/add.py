import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm

# create a 10 x 10 vertex mesh
xx, yy = np.meshgrid(np.linspace(0,9,10), np.linspace(0,9,10))

# create vertices for a rotated mesh (3D rotation matrix)
X =  xx
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
cset = ax2.contourf(X, Y, Z,1000)
for cnt, xx in enumerate(X_all):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    ax2.scatter(w[0], w[1],Z[w[1], w[0]],color=colors[t[cnt]])




plt.colorbar(cset)
plt.show()

