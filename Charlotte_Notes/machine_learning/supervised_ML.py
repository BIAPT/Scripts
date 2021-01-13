import matplotlib
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import metrics
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import random
import sys
sys.path.append('../')
#from dataimport.prepareDataset import *
from dataimport.prepareDataset_dPLI import *

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
Logistic Regression
################
'''
from sklearn.linear_model import LogisticRegression

cs=np.arange(1,5,0.1)
lr_accuracy_Base=[]
lr_accuracy_Anes=[]
lr_accuracy_Reco=[]

for c in cs:
    lr=LogisticRegression(random_state=0,penalty='l1',C=c)
    lr.fit(X_train_Base,Y_train_Base)
    P_lr=lr.predict(X_test_Base)
    lr_accuracy_Base.append(metrics.accuracy_score(Y_test_Base, P_lr))

    lr=LogisticRegression(random_state=0,penalty='l1',C=c)
    lr.fit(X_train_Anes,Y_train_Anes)
    P_lr=lr.predict(X_test_Anes)
    lr_accuracy_Anes.append(metrics.accuracy_score(Y_test_Anes, P_lr))

    lr=LogisticRegression(random_state=0,penalty='l1',C=c)
    lr.fit(X_train_Reco,Y_train_Reco)
    P_lr=lr.predict(X_test_Reco)
    lr_accuracy_Reco.append(metrics.accuracy_score(Y_test_Reco, P_lr))

#feat_importances_Reco = pd.Series(lr.coef_[0], index=X.columns)
#feat_importances_Reco.plot(kind='barh')


plt.plot(cs,lr_accuracy_Base)
plt.plot(cs,lr_accuracy_Anes)
plt.plot(cs,lr_accuracy_Reco)
plt.ylabel('Accuracy [%]')
plt.xlabel('C')
plt.title('Linear Regression')
plt.legend(['Baseline','Anesthesia','Recovery'])
plt.show()

cv_LR_Base=[]
cv_LR_Anes=[]
cv_LR_Reco=[]

FI_LR_Base=[]
FI_LR_Anes=[]
FI_LR_Reco=[]

for r in range(0,4):
    for c in range (0,4):
        tmp_X_test_Base=X_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_X_train_Base=X_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]
        tmp_Y_test_Base=Y_out_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_Y_train_Base=Y_out_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]

        tmp_X_test_Anes=X_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_X_train_Anes=X_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]
        tmp_Y_test_Anes=Y_out_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_Y_train_Anes=Y_out_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]

        tmp_X_test_Reco=X_Reco[(Y_ID_Reco == Part_reco[r]) | (Y_ID_Reco == Part_chro[c])]
        tmp_X_train_Reco=X_Reco[(Y_ID_Reco != Part_reco[r]) & (Y_ID_Reco != Part_chro[c])]
        tmp_Y_test_Reco=Y_out_Reco[(Y_ID_Reco == Part_reco[r]) | (Y_ID_Reco == Part_chro[c])]
        tmp_Y_train_Reco=Y_out_Reco[(Y_ID_Reco != Part_reco[r]) & (Y_ID_Reco != Part_chro[c])]

        lr = LogisticRegression(random_state=0, penalty='l1', C=4,max_iter=1000)
        lr.fit(tmp_X_train_Base, tmp_Y_train_Base)
        P_lr = lr.predict(tmp_X_test_Base)
        cv_LR_Base.append(metrics.accuracy_score(tmp_Y_test_Base, P_lr))
        FI_LR_Base.append(lr.coef_)

        lr = LogisticRegression(random_state=0, penalty='l1', C=4,max_iter=1000)
        lr.fit(tmp_X_train_Anes, tmp_Y_train_Anes)
        P_lr = lr.predict(tmp_X_test_Anes)
        cv_LR_Anes.append(metrics.accuracy_score(tmp_Y_test_Anes, P_lr))
        FI_LR_Anes.append(lr.coef_)

        lr = LogisticRegression(random_state=0, penalty='l1', C=4,max_iter=1000)
        lr.fit(tmp_X_train_Reco, tmp_Y_train_Reco)
        P_lr = lr.predict(tmp_X_test_Reco)
        cv_LR_Reco.append(metrics.accuracy_score(tmp_Y_test_Reco, P_lr))
        FI_LR_Reco.append(lr.coef_)


right_Anes = np.where(np.array(cv_LR_Anes) > 0.5)[0]
right_Base = np.where(np.array(cv_LR_Base) > 0.5)[0]
right_Reco = np.where(np.array(cv_LR_Reco) > 0.5)[0]

FI_LR_Base= list(FI_LR_Base[i] for i in right_Base)
FI_LR_Anes= list(FI_LR_Anes[i] for i in right_Anes)
FI_LR_Reco= list(FI_LR_Reco[i] for i in right_Reco)

feat_importances_Base_LR = pd.Series(abs(np.mean(FI_LR_Base[0:],axis=0)[0]), index=X.columns)
feat_importances_Anes_LR = pd.Series(abs(np.mean(FI_LR_Anes[0:],axis=0)[0]), index=X.columns)
feat_importances_Reco_LR = pd.Series(abs(np.mean(FI_LR_Reco[0:],axis=0)[0]), index=X.columns)

plt.subplot(121)
feat_importances_Base_LR.plot(kind='barh')
plt.title('LR_Baseline')
plt.subplot(122)
feat_importances_Anes_LR.plot(kind='barh',color='orange')
plt.title('Anesthesia')

#plt.subplot(133)
#feat_importances_Reco_LR.plot(kind='barh',color='green')
#plt.title('Recovery')



np.mean(cv_LR_Base)
np.std(cv_LR_Base)

np.mean(cv_LR_Anes)
np.std(cv_LR_Anes)

np.mean(cv_LR_Reco)
np.std(cv_LR_Reco)

plt.plot(cv_LR_Base)
plt.plot(cv_LR_Anes)
plt.plot(cv_LR_Reco)
plt.legend(['Baseline','Anesthesia','Recovery'])
plt.xlabel('cross validation')
plt.ylabel('accuracy')
plt.title('dPLI_CV')

'''
#################
SVM (sklearn)
################
'''
cs=np.arange(0.3,6,0.2)

svm_accuracy_Base=[]
svm_accuracy_Anes=[]
svm_accuracy_Reco=[]

for c in cs:
    svm_model = svm.LinearSVC(C=c, loss="hinge" , max_iter=15000)
    svm_model.fit(X_train_Base,Y_train_Base)
    P_lr=svm_model.predict(X_test_Base)
    svm_accuracy_Base.append(metrics.accuracy_score(Y_test_Base, P_lr))

    svm_model = svm.LinearSVC(C=c, loss="hinge", max_iter=15000)
    svm_model.fit(X_train_Anes,Y_train_Anes)
    P_lr=svm_model.predict(X_test_Anes)
    svm_accuracy_Anes.append(metrics.accuracy_score(Y_test_Anes, P_lr))

    svm_model = svm.LinearSVC(C=c, loss="hinge", max_iter=15000)
    svm_model.fit(X_train_Reco,Y_train_Reco)
    P_lr=svm_model.predict(X_test_Reco)
    svm_accuracy_Reco.append(metrics.accuracy_score(Y_test_Reco, P_lr))

plt.plot(cs,svm_accuracy_Base)
plt.plot(cs,svm_accuracy_Anes)
plt.plot(cs,svm_accuracy_Reco)
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.title('Support Vector Machine')
plt.legend(['Baseline','Anesthesia','Recovery'])
plt.show()


cv_SVM_Base=[]
cv_SVM_Anes=[]
cv_SVM_Reco=[]

FI_SVM_Base=[]
FI_SVM_Anes=[]
FI_SVM_Reco=[]


for r in range(0,4):
    for c in range (0,4):
        tmp_X_test_Base=X_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_X_train_Base=X_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]
        tmp_Y_test_Base=Y_out_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_Y_train_Base=Y_out_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]

        tmp_X_test_Anes=X_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_X_train_Anes=X_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]
        tmp_Y_test_Anes=Y_out_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_Y_train_Anes=Y_out_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]

        tmp_X_test_Reco=X_Reco[(Y_ID_Reco == Part_reco[r]) | (Y_ID_Reco == Part_chro[c])]
        tmp_X_train_Reco=X_Reco[(Y_ID_Reco != Part_reco[r]) & (Y_ID_Reco != Part_chro[c])]
        tmp_Y_test_Reco=Y_out_Reco[(Y_ID_Reco == Part_reco[r]) | (Y_ID_Reco == Part_chro[c])]
        tmp_Y_train_Reco=Y_out_Reco[(Y_ID_Reco != Part_reco[r]) & (Y_ID_Reco != Part_chro[c])]

        svm_model = svm.LinearSVC(C=4, loss="hinge", max_iter=15000)
        svm_model.fit(tmp_X_train_Base, tmp_Y_train_Base)
        P_lr = svm_model.predict(tmp_X_test_Base)
        cv_SVM_Base.append(metrics.accuracy_score(tmp_Y_test_Base, P_lr))
        FI_SVM_Base.append(svm_model.coef_.flatten())

        svm_model = svm.LinearSVC(C=4, loss="hinge", max_iter=15000)
        svm_model.fit(tmp_X_train_Anes, tmp_Y_train_Anes)
        P_lr = svm_model.predict(tmp_X_test_Anes)
        cv_SVM_Anes.append(metrics.accuracy_score(tmp_Y_test_Anes, P_lr))
        FI_SVM_Anes.append(svm_model.coef_.flatten())

        svm_model = svm.LinearSVC(C=4, loss="hinge", max_iter=15000)
        svm_model.fit(tmp_X_train_Reco, tmp_Y_train_Reco)
        P_lr = svm_model.predict(tmp_X_test_Reco)
        cv_SVM_Reco.append(metrics.accuracy_score(tmp_Y_test_Reco, P_lr))
        FI_SVM_Reco.append(svm_model.coef_.flatten())


np.mean(cv_SVM_Base)
np.std(cv_SVM_Base)

np.mean(cv_SVM_Anes)
np.std(cv_SVM_Anes)

np.mean(cv_SVM_Reco)
np.std(cv_SVM_Reco)

plt.plot(cv_SVM_Base)
plt.plot(cv_SVM_Anes)
plt.plot(cv_SVM_Reco)
plt.legend(['Baseline','Anesthesia','Recovery'])
plt.xlabel('cross validation')
plt.ylabel('accuracy')
plt.title('SVM')


right_Anes = np.where(np.array(cv_SVM_Anes) > 0.5)[0]
right_Base = np.where(np.array(cv_SVM_Base) > 0.5)[0]
right_Reco = np.where(np.array(cv_SVM_Reco) > 0.5)[0]


FI_SVM_Base=pd.DataFrame(FI_SVM_Base)
FI_SVM_Anes=pd.DataFrame(FI_SVM_Anes)
FI_SVM_Reco=pd.DataFrame(FI_SVM_Reco)

feat_importances_Base_SVM_b = pd.Series(np.array(abs(np.mean(FI_SVM_Base.iloc[right_Base,:],axis=0))), index=X.columns)
feat_importances_Anes_SVM_b = pd.Series(np.array(abs(np.mean(FI_SVM_Anes.iloc[right_Anes,:],axis=0))), index=X.columns)
feat_importances_Reco_SVM_b = pd.Series(np.array(abs(np.mean(FI_SVM_Reco.iloc[right_Reco,:],axis=0))), index=X.columns)

feat_importances_Base_SVM = pd.Series(np.array((np.mean(FI_SVM_Base.iloc[right_Base,:],axis=0))), index=X.columns)
feat_importances_Anes_SVM = pd.Series(np.array((np.mean(FI_SVM_Anes.iloc[right_Anes,:],axis=0))), index=X.columns)
feat_importances_Reco_SVM = pd.Series(np.array((np.mean(FI_SVM_Reco.iloc[right_Reco,:],axis=0))), index=X.columns)


plt.subplot(121)
feat_importances_Base_SVM_b.plot(kind='barh')
plt.title('SVM_Baseline')
plt.subplot(122)
feat_importances_Anes_SVM_b.plot(kind='barh',color='orange')
plt.title('Anesthesia')

#plt.subplot(133)
#feat_importances_Reco_SVM_b.plot(kind='barh',color='green')
#plt.title('Recovery')




plt.figure()
FC_Anes=np.mean(feat_importances_Anes_SVM_b[0:69])
FC_Base=np.mean(feat_importances_Base_SVM_b[0:69])
FP_Anes=np.mean(feat_importances_Anes_SVM_b[70:123])
FP_Base=np.mean(feat_importances_Base_SVM_b[70:123])
FO_Anes=np.mean(feat_importances_Anes_SVM_b[124:169])
FO_Base=np.mean(feat_importances_Base_SVM_b[124:169])
FT_Anes=np.mean(feat_importances_Anes_SVM_b[170:210])
FT_Base=np.mean(feat_importances_Base_SVM_b[170:210])
TO_Anes=np.mean(feat_importances_Anes_SVM_b[211:246])
TO_Base=np.mean(feat_importances_Base_SVM_b[211:246])
TC_Anes=np.mean(feat_importances_Anes_SVM_b[247:278])
TC_Base=np.mean(feat_importances_Base_SVM_b[247:278])
TP_Anes=np.mean(feat_importances_Anes_SVM_b[279:305])
TP_Base=np.mean(feat_importances_Base_SVM_b[279:305])
PO_Anes=np.mean(feat_importances_Anes_SVM_b[306:325])
PO_Base=np.mean(feat_importances_Base_SVM_b[306:325])
PC_Anes=np.mean(feat_importances_Anes_SVM_b[326:339])
PC_Base=np.mean(feat_importances_Base_SVM_b[326:339])
CO_Anes=np.mean(feat_importances_Anes_SVM_b[340:346])
CO_Base=np.mean(feat_importances_Base_SVM_b[340:346])


# set width of bar
barWidth = 0.3

# set height of bar
bars_A = [FC_Anes, FP_Anes, FO_Anes, FT_Anes, TO_Anes, TC_Anes, TP_Anes, PO_Anes, PC_Anes, CO_Anes]
bars_B = [FC_Base, FP_Base, FO_Base, FT_Base, TO_Base, TC_Base, TP_Base, PO_Base, PC_Base, CO_Base]

# Set position of bar on X axis
r1 = np.arange(len(bars_B))
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

# Create legend & Show graphic
plt.legend()
plt.show()


#plot single areas
feat_importances_Base_SVM_b[124:169].plot(kind='barh')
plt.title('Frontal-Occipital Feature Importance during Baseline')

plt.figure()
feat_importances_Base_SVM.nlargest(10).plot(kind='barh')
plt.figure()
feat_importances_Anes_SVM.nsmallest(20).plot(kind='barh',color = 'orange')

plt.figure()
plt.plot(feat_importances_Base_SVM)
plt.figure()
plt.plot(feat_importances_Anes_SVM)
plt.figure()
plt.plot(feat_importances_Reco_SVM)



"""
#########################################
GAUSSIAN NAIVE BAYES
#########################################
"""
from sklearn.naive_bayes import MultinomialNB

cs=np.arange(0.1,3.5,0.1)

gnb_accuracy_Base=[]
gnb_accuracy_Anes=[]
gnb_accuracy_Reco=[]

for a in cs:
    gnb = MultinomialNB(alpha=a)
    y_pred = gnb.fit(X_train_Base, Y_train_Base).predict(X_test_Base)
    gnb_accuracy_Base.append(metrics.accuracy_score(Y_test_Base, y_pred))

    gnb = MultinomialNB(alpha=a)
    y_pred = gnb.fit(X_train_Reco, Y_train_Reco).predict(X_test_Reco)
    gnb_accuracy_Reco.append(metrics.accuracy_score(Y_test_Reco, y_pred))

    gnb = MultinomialNB(alpha=a)
    y_pred = gnb.fit(X_train_Anes, Y_train_Anes).predict(X_test_Anes)
    gnb_accuracy_Anes.append(metrics.accuracy_score(Y_test_Anes, y_pred))


plt.plot(cs,gnb_accuracy_Base)
plt.plot(cs,gnb_accuracy_Anes)
plt.plot(cs,gnb_accuracy_Reco)
plt.ylabel('Accuracy')
plt.xlabel('alpha')
plt.title('Naive Bayes')
plt.legend(['Baseline','Anesthesia','Recovery'])
plt.show()

cv_gnb_Base=[]
cv_gnb_Anes=[]
cv_gnb_Reco=[]

for r in range(0,4):
    for c in range (0,4):
        tmp_X_test_Base=X_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_X_train_Base=X_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]
        tmp_Y_test_Base=Y_out_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_Y_train_Base=Y_out_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]

        tmp_X_test_Anes=X_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_X_train_Anes=X_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]
        tmp_Y_test_Anes=Y_out_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_Y_train_Anes=Y_out_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]

        tmp_X_test_Reco=X_Reco[(Y_ID_Reco == Part_reco[r]) | (Y_ID_Reco == Part_chro[c])]
        tmp_X_train_Reco=X_Reco[(Y_ID_Reco != Part_reco[r]) & (Y_ID_Reco != Part_chro[c])]
        tmp_Y_test_Reco=Y_out_Reco[(Y_ID_Reco == Part_reco[r]) | (Y_ID_Reco == Part_chro[c])]
        tmp_Y_train_Reco=Y_out_Reco[(Y_ID_Reco != Part_reco[r]) & (Y_ID_Reco != Part_chro[c])]

        gnb = MultinomialNB(alpha=0.1)
        gnb.fit(tmp_X_train_Base, tmp_Y_train_Base)
        P_lr = gnb.predict(tmp_X_test_Base)
        cv_gnb_Base.append(metrics.accuracy_score(tmp_Y_test_Base, P_lr))

        gnb = MultinomialNB(alpha=0.3)
        gnb.fit(tmp_X_train_Anes, tmp_Y_train_Anes)
        P_lr = gnb.predict(tmp_X_test_Anes)
        cv_gnb_Anes.append(metrics.accuracy_score(tmp_Y_test_Anes, P_lr))

        gnb = MultinomialNB(alpha=0.1)
        gnb.fit(tmp_X_train_Reco, tmp_Y_train_Reco)
        P_lr = gnb.predict(tmp_X_test_Reco)
        cv_gnb_Reco.append(metrics.accuracy_score(tmp_Y_test_Reco, P_lr))


np.mean(cv_gnb_Base)
np.std(cv_gnb_Base)

np.mean(cv_gnb_Anes)
np.std(cv_gnb_Anes)

np.mean(cv_gnb_Reco)
np.std(cv_gnb_Reco)

plt.plot(cv_gnb_Base)
plt.plot(cv_gnb_Anes)
plt.plot(cv_gnb_Reco)
plt.legend(['Baseline','Anesthesia','Recovery'])
plt.xlabel('cross validation')
plt.ylabel('accuracy')





# Decision Tree
from sklearn import tree
import graphviz
import os
#os.environ["PATH"] += os.pathsep + 'C:/Users/User/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz/'


cv_DT_accuracy = []


for r in range(0, 4):
    for c in range(0, 4):
        tmp_X_test_Base = X_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_X_train_Base = X_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]
        tmp_Y_test_Base = Y_out_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_Y_train_Base = Y_out_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]

        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(tmp_X_train_Base,tmp_Y_train_Base)
        P=clf.predict(tmp_X_test_Base)
        acc=metrics.accuracy_score(tmp_Y_test_Base, P)
        cv_DT_accuracy.append(acc)

        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=names, class_names=['Chronic', 'recovered'],
                                        filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(Part_reco[r]+'_'+Part_chro[c]+'_'+str(acc*100))



np.mean(cv_DT_accuracy)
np.std(cv_DT_accuracy)
plt.plot(cv_DT_accuracy)


cv_DT_accuracy = []
for r in range(0, 4):
    for c in range(0, 4):
        tmp_X_test_Anes = X_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_X_train_Anes = X_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]
        tmp_Y_test_Anes = Y_out_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_Y_train_Anes = Y_out_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]

        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(tmp_X_train_Anes,tmp_Y_train_Anes)
        P=clf.predict(tmp_X_test_Anes)
        acc=metrics.accuracy_score(tmp_Y_test_Anes, P)
        cv_DT_accuracy.append(acc)

        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=names, class_names=['Chronic', 'recovered'],
                                        filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(Part_reco[r]+'_'+Part_chro[c]+'_'+str(acc*100))



np.mean(cv_DT_accuracy)
np.std(cv_DT_accuracy)
plt.plot(cv_DT_accuracy)
