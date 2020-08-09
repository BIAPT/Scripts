import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')
import numpy as np
from LSTM import data_LSTM

part_reco = ['19', '02', '20', '09']
part_chro = ['13', '18', '05', '11', '22', '12', '10']
#part_reco = ['09']
#part_chro = [ '10']
data = pd.read_pickle('data/WholeBrain_wPLI_10_1_alpha.pickle')
X=data.query("ID == '19'")
X=X.query("Phase == 'Anes'")
Y_Phase=X['Phase']
X=X.iloc[:,4:]

labels=np.zeros(len(Y_Phase))
labels[Y_Phase=='Anes']=1
labels[Y_Phase=='Reco']=2


from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, perplexity=50, random_state=2020)
tsne_model.fit(X)
score=tsne_model.fit_transform(X,labels)

component1 = score[:,0]
component2 = score[:,1]

plt.figure()
plt.scatter(x=component1[labels==0], y=component2[labels==0], c='blue')
plt.scatter(x=component1[labels==1], y=component2[labels==1], c='orange')
plt.scatter(x=component1[labels==2], y=component2[labels==2], c='green')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('T-SNE')
plt.legend(['Baseline','Anesthesia','Recovery'])
plt.show()



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X3 = pca.transform(X)

plt.figure()
plt.scatter(x=X3[:,0], y=X3[:,1], c=labels)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA')
plt.show()

pca.components_.shape