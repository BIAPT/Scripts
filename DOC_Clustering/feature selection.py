import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import metrics
import matplotlib.pyplot as plt
import random
from prepareDataset import *

X=X.iloc[:,empty==0]
X_Anes=X_Anes.iloc[:,empty==0]
X_Base=X_Base.iloc[:,empty==0]
X_Reco=X_Reco.iloc[:,empty==0]

random.seed(141)

#X_train,X_test,Y_train,Y_test=train_test_split(X,Y_out,Y_ID,random_state=0,test_size=0.3)
X_train_Base,X_test_Base,Y_train_Base,Y_test_Base,Y_ID_Base_train,Y_ID_Base_test=train_test_split(X_Base,Y_out_Base,Y_ID_Base,random_state=0,test_size=0.3)
X_train_Reco,X_test_Reco,Y_train_Reco,Y_test_Reco,Y_ID_Reco_train,Y_ID_Reco_test=train_test_split(X_Reco,Y_out_Reco,Y_ID_Reco,random_state=0,test_size=0.3)
X_train_Anes,X_test_Anes,Y_train_Anes,Y_test_Anes,Y_ID_Anes_train,Y_ID_Anes_test=train_test_split(X_Anes,Y_out_Anes,Y_ID_Anes,random_state=0,test_size=0.3)

'''
#################
Extra Trees Classifier
################
'''
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


cv_FS_Base=[]
cv_FS_Anes=[]
cv_FS_Reco=[]
FI_Base=[]
FI_Anes=[]
FI_Reco=[]



for r in range(0,4):
    for c in range (0,4):
        tmp_X_test_Base = X_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_X_train_Base = X_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]
        tmp_Y_test_Base = Y_out_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_Y_train_Base = Y_out_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]

        tmp_X_test_Anes = X_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_X_train_Anes = X_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]
        tmp_Y_test_Anes = Y_out_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_Y_train_Anes = Y_out_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]

        tmp_X_test_Reco = X_Reco[(Y_ID_Reco == Part_reco[r]) | (Y_ID_Reco == Part_chro[c])]
        tmp_X_train_Reco = X_Reco[(Y_ID_Reco != Part_reco[r]) & (Y_ID_Reco != Part_chro[c])]
        tmp_Y_test_Reco = Y_out_Reco[(Y_ID_Reco == Part_reco[r]) | (Y_ID_Reco == Part_chro[c])]
        tmp_Y_train_Reco = Y_out_Reco[(Y_ID_Reco != Part_reco[r]) & (Y_ID_Reco != Part_chro[c])]

        model = ExtraTreesClassifier()
        model.fit(tmp_X_train_Base,tmp_Y_train_Base)
        P2=model.predict(tmp_X_test_Base)
        cv_FS_Base.append(metrics.accuracy_score(tmp_Y_test_Base, P2))
        FI_Base.append([model.feature_importances_])

        model = ExtraTreesClassifier()
        model.fit(tmp_X_train_Anes,tmp_Y_train_Anes)
        P2=model.predict(tmp_X_test_Anes)
        cv_FS_Anes.append(metrics.accuracy_score(tmp_Y_test_Anes, P2))
        FI_Anes.append([model.feature_importances_])

        model = ExtraTreesClassifier()
        model.fit(tmp_X_train_Reco,tmp_Y_train_Reco)
        P2=model.predict(tmp_X_test_Reco)
        cv_FS_Reco.append(metrics.accuracy_score(tmp_Y_test_Reco, P2))
        FI_Reco.append([model.feature_importances_])


plt.plot(cv_FS_Base)
np.mean(cv_FS_Base)
np.std(cv_FS_Base)

np.mean(cv_FS_Anes)
np.std(cv_FS_Anes)

np.mean(cv_FS_Reco)
np.std(cv_FS_Reco)


#plot graph of feature importances for better visualization
feat_importances_Base = pd.Series(np.mean(FI_Base[0:16],axis=0)[0], index=X.columns)
feat_importances_Anes = pd.Series(np.mean(FI_Anes[0:16],axis=0)[0], index=X.columns)
feat_importances_Reco = pd.Series(np.mean(FI_Reco[0:16],axis=0)[0], index=X.columns)

feat_importances_Base.plot(kind='barh',color='blue')
feat_importances_Anes.plot(kind='barh',color='orange')
feat_importances_Reco.plot(kind='barh',color='green')

feat_importances_Anes.nlargest(30).plot(kind='barh')
plt.show()
plt.title('Feature Importance Baseline')

