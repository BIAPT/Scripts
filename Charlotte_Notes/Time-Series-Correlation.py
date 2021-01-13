import scipy.io
import extract_features
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

"""
################################
        LOAD DATA (not available on gitHub)
################################
"""

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_anes_avg.mat')
data_anes_avg = mat['result_wpli_anes_avg']
X_anes_avg=extract_features.extract_features(data_anes_avg)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_rest_avg.mat')
data_rest_avg = mat['result_wpli_rest_avg']
X_rest_avg=extract_features.extract_features(data_rest_avg)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_anes_step.mat')
data_anes_step = mat['result_wpli_anes_step']
# extract 5460 features
X_anes_step=extract_features.extract_features(data_anes_step)
#X_anes_step_diff = extract_features.get_difference(X_anes_step)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_rest_step.mat')
data_rest_step = mat['result_wpli_rest_step']
# extract 5460 features
X_rest_step=extract_features.extract_features(data_rest_step)
#X_rest_step_diff = extract_features.get_difference(X_rest_step)

max_len=min(len(X_rest_step),len(X_anes_step))
nr_features=X_rest_step.shape[1]

"""
################################
        CORRELATION ANALYSIS
################################
"""

import scipy.stats as stats
import pandas as pd

# create empty correlation matrix
anes_cor=np.zeros([nr_features,nr_features])
rest_cor=np.zeros([nr_features,nr_features])

# transform to dataframe
X_anes_step=pd.DataFrame(X_anes_step)
X_rest_step=pd.DataFrame(X_rest_step)

for i in range(0,nr_features):
    for j in range(0,nr_features):
        r, p = stats.pearsonr(X_rest_step.iloc[0:max_len,i], X_rest_step.iloc[0:max_len,j])
        if p < 0.05:
            rest_cor[i,j]=r
        elif p <= 0.05:
            rest_cor[i, j] = 0

plt.imshow(rest_cor)

#np.savetxt('rest_corr.txt', rest_cor, fmt='%s',delimiter=';')
np.save('rest_corr',rest_cor)

rest_cor=np.load('rest_corr.npy')
plt.imshow(rest_cor,cmap='seismic')
plt.legend()
min(rest_cor,'all')

#https://plot.ly/python/v3/network-graphs/
#https://www.youtube.com/watch?v=NEaUSP4YerM
#https://towardsdatascience.com/python-interactive-network-visualization-using-networkx-plotly-and-dash-e44749161ed7

bin_cor=np.load('rest_corr.npy')

for i in range(0,bin_cor.shape[1]):
    for a in range(0,i):
        bin_cor[i,a]=0


bin_cor[np.where(bin_cor==1)[0],np.where(bin_cor==1)[1]]=0
bin_cor[np.where(bin_cor<0)[0],np.where(bin_cor<0)[1]]=0
bin_cor[np.where(bin_cor>=0.8)[0],np.where(bin_cor>=0.8)[1]]=1

len(np.where(bin_cor==1)[0])
plt.imshow(bin_cor)

bin_cor_df=pd.DataFrame(bin_cor)

# Transform it in a links data frame (3 columns only):
links = bin_cor_df.stack().reset_index()
links.columns = ['source', 'target', 'weight']
links.shape

# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
links_filtered = links.loc[(links['weight'] > 0) & (links['source'] != links['target'])]
links_filtered.shape


import networkx as nx
# Build your graph
G = nx.from_pandas_edgelist(links)
G_f = nx.from_pandas_edgelist(links_filtered)

print(nx.info(G_f))
print(nx.info(G))

selected=list(G_f.nodes)
features=list(np.arange(0,5460))

eoe=pd.DataFrame(set(features)-set(selected))
len(eoe)

#nx.write_gexf(G, "bin_rest_cor_0.8.gexf")

# get disconnected subgraphs
d = list(nx.connected_component_subgraphs(G_f))
# d contains disconnected subgraphs
# d[0] contains the biggest subgraph
len(d)
print(nx.info(d[0]))
print(nx.info(d[1]))

degree_centrality = nx.degree_centrality(G_f)

names = ['id','data']
formats = ['float','float']
dtype = dict(names = names, formats=formats)
array = np.array(list(degree_centrality.items()), dtype=dtype)

plt.plot(array['data'])

print(degree_centrality)
assert degree_centrality["E"] == 4/5





bin_cor.shape
s=sum(bin_cor,1)-1
m=np.mean(bin_cor,1)

plt.plot(m)

min(s[s<=0])












