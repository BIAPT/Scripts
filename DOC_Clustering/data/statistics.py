import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import model_selection, naive_bayes, svm
from sklearn.model_selection import permutation_test_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import metrics
import random
import seaborn as sns
from pandas import plotting

# %%

random.seed(1)

# %%

Part_chro = ['13', '22', '10', '18']
Part_reco = ['19', '20', '02', '09']

# %%
areas = ['FC', 'FP', 'FO', 'FT', 'TO', 'TC', 'TP', 'PO', 'PC', 'CO', 'FF', 'CC', 'PP', 'TT', 'OO']

data = pd.read_pickle('data/NEW_wPLI_all_10_1_left.pickle')

# %%
# chronic 0
data.insert(0, 'outcome', "0")

# %%
for p in Part_reco:
    data.iloc[np.where(data['ID'] == p)[0], 0] = 1

# %%
grouped_data = data.groupby('outcome')

# %%
plotting.scatter_matrix(data[areas], c=data['outcome'])
fig = plt.gcf()
plt.show()

data_Base=data[(data['Phase'] == 'Base')]
data_Anes=data[(data['Phase'] == 'Anes')]
data_Reco=data[(data['Phase'] == 'Reco')]

plotting.scatter_matrix(data_Anes[areas],c=(data_Anes['outcome']==1))
fig=plt.gcf()
plt.title("violet: chronic   yellow: recovered")
plt.show()

Y_ID = data.iloc[:, 2]

data_chro = data[(Y_ID == '13') | (Y_ID == '22') | (Y_ID == '10') | (Y_ID == '18')]
data_reco = data[(Y_ID == '19') | (Y_ID == '20') | (Y_ID == '02') | (Y_ID == '09')]

data_Base_c=data_chro[(data_chro['Phase'] == 'Base')]
data_Anes_c=data_chro[(data_chro['Phase'] == 'Anes')]
data_Reco_c=data_chro[(data_chro['Phase'] == 'Reco')]

data_Base_r=data_reco[(data_reco['Phase'] == 'Base')]
data_Anes_r=data_reco[(data_reco['Phase'] == 'Anes')]
data_Reco_r=data_reco[(data_reco['Phase'] == 'Reco')]

corrB_c = data_Base_c[areas].corr()
corrA_c = data_Anes_c[areas].corr()
corrR_c = data_Reco_c[areas].corr()

corrB_r = data_Base_r[areas].corr()
corrA_r = data_Anes_r[areas].corr()
corrR_r = data_Reco_r[areas].corr()

figure =plt.figure()
plt.subplot(231)
sns.heatmap(corrB_c,vmin=0, vmax=1)
plt.ylabel('chronic')
plt.subplot(232)
sns.heatmap(corrA_c,vmin=0, vmax=1)
plt.subplot(233)
sns.heatmap(corrR_c,vmin=0, vmax=1)
plt.subplot(234)
plt.ylabel('recovered')
plt.xlabel("BAseline")
sns.heatmap(corrB_r,vmin=0, vmax=1)
plt.subplot(235)
plt.xlabel("Anesthesia")
sns.heatmap(corrA_r,vmin=0, vmax=1)
plt.subplot(236)
plt.xlabel("Recovery")
sns.heatmap(corrR_r,vmin=0, vmax=1)

figure =plt.figure()
plt.subplot(211)
sns.heatmap(corrA_c-corrB_c,vmin=0, vmax=1)
plt.ylabel('chronic')
plt.subplot(212)
sns.heatmap(corrA_r-corrB_r,vmin=0, vmax=1)