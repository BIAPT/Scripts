import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import metrics
import matplotlib.pyplot as plt
import random
from sklearn import model_selection, naive_bayes, svm
from prepareDataset import *

X=X.iloc[:,empty==0]
X_Anes=X_Anes.iloc[:,empty==0]
X_Base=X_Base.iloc[:,empty==0]
X_Reco=X_Reco.iloc[:,empty==0]

X_B_A=np.zeros((217,2*347))
Y_B_A=np.zeros((217,5))
Y_B_A=pd.DataFrame(Y_B_A)


for i in range(0,data_Base.shape[0]):
    pB=data_Base.iloc[i,2]
    tB=data_Base.iloc[i,4]
    oB=data_Base.iloc[i,0]
    XB=data_Base.iloc[i,5:]
    XB=XB.iloc[empty==0]
    a=np.where((data_Anes.iloc[:,2]==pB) & (data_Anes.iloc[:,4]==tB))[0][0]
    pA = data_Anes.iloc[a, 2]
    tA = data_Anes.iloc[a, 4]
    XA = data_Anes.iloc[a,5:]
    XA = XA.iloc[empty == 0]
    XBA=XB.append(XA)
    X_B_A[i]=XBA
    Y_B_A.iloc[i,0]=oB
    Y_B_A.iloc[i,1]=pB
    Y_B_A.iloc[i,2]=tB
    Y_B_A.iloc[i,3]=pA
    Y_B_A.iloc[i,4]=tA

random.seed(141)

X_train_BA,X_test_BA,Y_train_BA,Y_test_BA=train_test_split(X_B_A,Y_B_A[0],random_state=0,test_size=0.3)

cs=np.arange(0.1,2,0.1)
svm_accuracy_BA=[]

for c in cs:
    svm_model = svm.LinearSVC(C=c, loss="hinge" , max_iter=15000)
    svm_model.fit(X_train_BA,Y_train_BA)
    P_lr=svm_model.predict(X_test_BA)
    svm_accuracy_BA.append(metrics.accuracy_score(Y_test_BA, P_lr))

plt.plot(cs,svm_accuracy_BA)
max(svm_accuracy_BA)


cv_svm_accuracy_BA = []
for r in range(0,4):
    for c in range (0,4):
        tmp_X_test_BA=X_B_A[(Y_B_A[1] == Part_reco[r]) | (Y_B_A[1] == Part_chro[c])]
        tmp_X_train_BA=X_B_A[(Y_B_A[1] != Part_reco[r]) & (Y_B_A[1] != Part_chro[c])]
        tmp_Y_test_BA=Y_B_A[(Y_B_A[1] == Part_reco[r]) | (Y_B_A[1] == Part_chro[c])]
        tmp_Y_train_BA=Y_B_A[(Y_B_A[1] != Part_reco[r]) & (Y_B_A[1] != Part_chro[c])]
        tmp_Y_test_BA=tmp_Y_test_BA[0]
        tmp_Y_train_BA=tmp_Y_train_BA[0]

        svm_model = svm.LinearSVC(C=1, loss="hinge", max_iter=15000)
        svm_model.fit(tmp_X_train_BA, tmp_Y_train_BA)
        P_lr = svm_model.predict(tmp_X_test_BA)
        cv_svm_accuracy_BA.append(metrics.accuracy_score(tmp_Y_test_BA, P_lr))


plt.plot(cv_svm_accuracy_BA)
np.mean(cv_svm_accuracy_BA)
np.std(cv_svm_accuracy_BA)


cv_svm_accuracy_BA = []
FI_SVM_BA=[]
for r in range(0, 4):
    for c in range(0, 4):
        tmp_X_test_BA = X_B_A[(Y_B_A[1] == Part_reco[r]) | (Y_B_A[1] == Part_chro[c])]
        tmp_X_train_BA = X_B_A[(Y_B_A[1] != Part_reco[r]) & (Y_B_A[1] != Part_chro[c])]
        tmp_Y_test_BA = Y_B_A[(Y_B_A[1] == Part_reco[r]) | (Y_B_A[1] == Part_chro[c])]
        tmp_Y_train_BA = Y_B_A[(Y_B_A[1] != Part_reco[r]) & (Y_B_A[1] != Part_chro[c])]
        tmp_Y_test_BA = tmp_Y_test_BA[0]
        tmp_Y_train_BA = tmp_Y_train_BA[0]

        svm_model = svm.LinearSVC(C=1, loss="hinge", max_iter=15000)
        svm_model.fit(tmp_X_train_BA, tmp_Y_train_BA)
        P_lr = svm_model.predict(tmp_X_test_BA)
        cv_svm_accuracy_BA.append(metrics.accuracy_score(tmp_Y_test_BA, P_lr))
        FI_SVM_BA.append(svm_model.coef_.flatten())

for r in range(0, 4):
    for c in range(0, 4):
        print(Part_reco[r]+"  "+Part_chro[c])

plt.plot(cv_svm_accuracy_BA)
np.mean(cv_svm_accuracy_BA)
np.std(cv_svm_accuracy_BA)

right_BA = np.where(np.array(cv_svm_accuracy_BA) > 0.55)[0]
FI_SVM_BA=pd.DataFrame(FI_SVM_BA)


names='B_'+X_Base.columns
names=names.append('A_'+X_Anes.columns)
feat_importances_BA_SVM_b = pd.Series(np.array(abs(np.mean(FI_SVM_BA.iloc[right_BA,:],axis=0))), index=names)


feat_importances_BA_SVM_b.nlargest(30).plot(kind='barh')

plt.figure()
FC_Anes=np.mean(feat_importances_BA_SVM_b[0+347:70+347])
FC_Base=np.mean(feat_importances_BA_SVM_b[0:70])
FP_Anes=np.mean(feat_importances_BA_SVM_b[71+347:123+347])
FP_Base=np.mean(feat_importances_BA_SVM_b[71:123])
FO_Anes=np.mean(feat_importances_BA_SVM_b[124+347:169+347])
FO_Base=np.mean(feat_importances_BA_SVM_b[124:169])
FT_Anes=np.mean(feat_importances_BA_SVM_b[170+347:210+347])
FT_Base=np.mean(feat_importances_BA_SVM_b[170:210])
TO_Anes=np.mean(feat_importances_BA_SVM_b[211+347:246+347])
TO_Base=np.mean(feat_importances_BA_SVM_b[211:246])
TC_Anes=np.mean(feat_importances_BA_SVM_b[247+347:278+347])
TC_Base=np.mean(feat_importances_BA_SVM_b[247:278])
TP_Anes=np.mean(feat_importances_BA_SVM_b[279+347:305+347])
TP_Base=np.mean(feat_importances_BA_SVM_b[279:305])
PO_Anes=np.mean(feat_importances_BA_SVM_b[306+347:325+347])
PO_Base=np.mean(feat_importances_BA_SVM_b[306:325])
PC_Anes=np.mean(feat_importances_BA_SVM_b[326+347:339+347])
PC_Base=np.mean(feat_importances_BA_SVM_b[326:339])
CO_Anes=np.mean(feat_importances_BA_SVM_b[340+347:346+347])
CO_Base=np.mean(feat_importances_BA_SVM_b[340:346])


feat_importances_BA_SVM_b.nlargest(10).plot(kind='barh')


# set width of bar
barWidth = 0.25

# set height of bar
bars_A = [FC_Anes, FP_Anes, FO_Anes, FT_Anes, TO_Anes, TC_Anes, TP_Anes, PO_Anes, PC_Anes, CO_Anes]
bars_B = [FC_Base, FP_Base, FO_Base, FT_Base, TO_Base, TC_Base, TP_Base, PO_Base, PC_Base, CO_Base]

# Set position of bar on X axis
r1 = np.arange(len(bars_A))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars_B,  width=barWidth, label='Baseline')
plt.bar(r2, bars_A,  width=barWidth, label='Anesthesia')

# Add xticks on the middle of the group bars
plt.xlabel('wPLI region', fontweight='bold')
plt.ylabel('feature importance', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_A))], ['FC', 'FP', 'FO', 'FT', 'TO', 'TC', 'TP', 'PO','PC', 'CO'])
plt.legend()
plt.title('Combined Data Feature Importance')


# Decision Tree
from sklearn import tree
import graphviz
import os
#os.environ["PATH"] += os.pathsep + 'C:/Users/User/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz/'


cv_DT_accuracy_BA = []


for r in range(0, 4):
    for c in range(0, 4):
        tmp_X_test_BA = X_B_A[(Y_B_A[1] == Part_reco[r]) | (Y_B_A[1] == Part_chro[c])]
        tmp_X_train_BA = X_B_A[(Y_B_A[1] != Part_reco[r]) & (Y_B_A[1] != Part_chro[c])]
        tmp_Y_test_BA = Y_B_A[(Y_B_A[1] == Part_reco[r]) | (Y_B_A[1] == Part_chro[c])]
        tmp_Y_train_BA = Y_B_A[(Y_B_A[1] != Part_reco[r]) & (Y_B_A[1] != Part_chro[c])]
        tmp_Y_test_BA = tmp_Y_test_BA[0]
        tmp_Y_train_BA = tmp_Y_train_BA[0]

        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(tmp_X_train_BA,tmp_Y_train_BA)
        P=clf.predict(tmp_X_test_BA)
        acc=metrics.accuracy_score(tmp_Y_test_BA, P)
        cv_DT_accuracy_BA.append(acc)

        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=names, class_names=['Chronic', 'recovered'],
                                        filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(Part_reco[r]+'_'+Part_chro[c]+'_'+str(acc*100))


np.mean(cv_DT_accuracy_BA)
plt.plot(cv_DT_accuracy_BA)

'''dot_data = tree.export_graphviz(clf, out_file=None,feature_names=names,class_names=['Chronic','recovered'],
                                filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph.view()
'''
