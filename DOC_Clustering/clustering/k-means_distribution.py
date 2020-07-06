import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.cluster import KMeans

# Parameters
Part_chro=['13','22','10', '18','05','12','11']
Part_reco=['19','20','02','09']
Phase=['Base','Anes']

for p in Phase:
    name="K_Means_Distribution_"+p+".pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(name)
    data = pd.read_pickle('data/WholeBrain_wPLI_10_1_alpha.pickle')
    data=data.query("Phase=='{}'".format(p))
    X= data.iloc[:,4:]
    Y_ID=data.iloc[:,1]
    Y_out=np.zeros(len(Y_ID))
    Y_out[(Y_ID == '19')|(Y_ID == '20')|(Y_ID == '02')|(Y_ID == '09')]=1

    Ks=[2,3,4,5,6,7]

    for K in Ks:
        kmeans = KMeans(n_clusters=K, max_iter=1000, n_init=100)
        kmeans.fit(X)
        Pred = kmeans.predict(X)

        Pred_reco=Pred[Y_out==1]
        Pred_nonr=Pred[Y_out==0]

        labels=[]
        sizes_reco=[]
        sizes_nonr=[]

        for i in range(K):
            labels.append('cluster '+str(i))
            sizes_reco.append(list(Pred_reco).count(i))
            sizes_nonr.append(list(Pred_nonr).count(i))

        Figure=plt.figure()
        plt.subplot(1,2,1)
        plt.pie(sizes_reco, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.title('recovered')
        plt.subplot(1,2,2)
        plt.pie(sizes_nonr, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.title('non-recovered')

        pdf.savefig(Figure)
        plt.close()

    pdf.close()


data = pd.read_pickle('data/WholeBrain_wPLI_10_1_alpha.pickle')
data=data.query("Phase=='Base'")
X= data.iloc[:,4:]
Y_ID=data.iloc[:,1]
Y_out=np.zeros(len(Y_ID))
Y_out[(Y_ID == '19')|(Y_ID == '20')|(Y_ID == '02')|(Y_ID == '09')]=1

kmeans = KMeans(n_clusters=4, max_iter=1000, n_init=100)
kmeans.fit(X)
Pred = kmeans.predict(X)

plt.plot(range(len(Y_ID[Pred==3])),Y_ID[Pred==3])
plt.title('Baseline_k= Cluster 2')

plt.plot(Pred[Y_ID=='02'])

Pred_reco=Pred[Y_out==1]
Pred_nonr=Pred[Y_out==0]

len(Pred_reco)
