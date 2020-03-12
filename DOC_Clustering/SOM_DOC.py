"""
   SOM
"""
from minisom import MiniSom
#https://github.com/JustGlowing/minisom/blob/master/examples/PoemsAnalysis.ipynb

map_dim = 10
som = MiniSom(map_dim, map_dim, 5460, sigma=1.0, random_seed=1)
#som.random_weights_init(W)
som.train_batch(X_all, num_iteration=len(X_all)*500, verbose=True)


plt.figure(figsize=(10, 10))
# Plotting the response for each pattern in the iris dataset
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
#plt.colorbar()

t = np.zeros(len(Y_all), dtype=int)
t[Y_all == 0] = 0
t[Y_all == 1] = 1

# use different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']
for cnt, xx in enumerate(X_all):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]], markerfacecolor='None',
             markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
plt.colorbar()
plt.axis([0, 7, 0, 7])
plt.savefig('resulting_images/som_iris.png')
plt.show()

