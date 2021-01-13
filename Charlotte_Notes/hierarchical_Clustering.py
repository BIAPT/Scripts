import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from load_data import *
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


#https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
#https://towardsdatascience.com/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
# ward minimizes variance between clusters

cluster.fit_predict(X_all)
print(cluster.labels_)

fig= plt.figure()
plot_dendrogram(cluster, labels=cluster.labels_)

import scipy.cluster.hierarchy as hc
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms_t")
# without transpose it is classifying the time points,
Y2 = hc.linkage(np.transpose(X_all)) # output electrode-order, features
Y3 = hc.linkage(X_all) # outputs the time order

Z2 = hc.dendrogram(Y2)
index2 = Z2['leaves']
len(index2)
plt.figure()
plt.plot(index2)

from pylab import rcParams
rcParams['figure.figsize'] = 100, 9
rcParams['axes.labelsize'] = "large"
rcParams['font.size']= 10
plt.figure(figsize=(10, 7))
plt.title("Feature Dendograms")
hc.dendrogram(Y2)

Z3 = hc.dendrogram(Y3)
index3 = Z3['leaves']
len(index3)

X_new=X_all[:,index2]
plt.figure(figsize=(5, 5))
plt.imshow(X_all)
plt.figure(figsize=(5, 5))
plt.imshow(X_new[index3,:])



plt.plot(index3)
plt.figure()
hc.dendrogram(Y3)

hc.dendrogram(Y2)
hc.dendrogram(Y3)

len(Y_all)
sortedY=Y_all[index3]

X_all_time_order = X_all[index,:]
fig= plt.figure()
plt.plot(sortedY)
plt.imshow(X_all_time_order,cmap='jet')


X_all_time_order.shape
X_all.shape
################  Default Methods ################
row_method = 'average'
column_method = 'single'
row_metric = 'cityblock'
column_metric = 'euclidean'
color_gradient = 'jet'

dend = shc.dendrogram(shc.linkage(X_anes_step, method='ward',metric=column_metric))
#http://datanongrata.com/2019/04/27/67/




import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X_all)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

from scipy.cluster import hierarchy

ZF = hierarchy.linkage(np.transpose(X_all), 'single')

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

#https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

plt.figure()
hierarchy.dendrogram(ZF,truncate_mode='lastp',
                     p=4,
                     leaf_rotation=90.,
                     leaf_font_size=12.,
                     show_contracted=True)


plt.figure(figsize=(20,10))
fancy_dendrogram(
    ZF,
    truncate_mode='lastp',
    p=3000,
    leaf_rotation=90.,
    leaf_font_size=2.,
    show_contracted=True,
    annotate_above=40,
    max_d=170,
)
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single')
pred=cluster.fit_predict(X_all)
plt.plot(Z_all)
from sklearn.metrics import accuracy_score
accuracy_score(pred,Y_all)

