import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import metrics
import matplotlib.pyplot as plt
import random
from prepareDataset import *

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

cs=np.arange(1,5.6,0.05)
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

        lr = LogisticRegression(random_state=0, penalty='l1', C=5)
        lr.fit(tmp_X_train_Base, tmp_Y_train_Base)
        P_lr = lr.predict(tmp_X_test_Base)
        cv_LR_Base.append(metrics.accuracy_score(tmp_Y_test_Base, P_lr))

        lr = LogisticRegression(random_state=0, penalty='l1', C=1.8)
        lr.fit(tmp_X_train_Anes, tmp_Y_train_Anes)
        P_lr = lr.predict(tmp_X_test_Anes)
        cv_LR_Anes.append(metrics.accuracy_score(tmp_Y_test_Anes, P_lr))

        lr = LogisticRegression(random_state=0, penalty='l1', C=3.9)
        lr.fit(tmp_X_train_Reco, tmp_Y_train_Reco)
        P_lr = lr.predict(tmp_X_test_Reco)
        cv_LR_Reco.append(metrics.accuracy_score(tmp_Y_test_Reco, P_lr))


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

'''
#################
SVM (sklearn)
################
'''
cs=np.arange(0.3,10,0.1)

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

        svm_model = svm.LinearSVC(C=5, loss="hinge", max_iter=15000)
        svm_model.fit(tmp_X_train_Base, tmp_Y_train_Base)
        P_lr = svm_model.predict(tmp_X_test_Base)
        cv_SVM_Base.append(metrics.accuracy_score(tmp_Y_test_Base, P_lr))

        svm_model = svm.LinearSVC(C=5, loss="hinge", max_iter=15000)
        svm_model.fit(tmp_X_train_Anes, tmp_Y_train_Anes)
        P_lr = svm_model.predict(tmp_X_test_Anes)
        cv_SVM_Anes.append(metrics.accuracy_score(tmp_Y_test_Anes, P_lr))

        svm_model = svm.LinearSVC(C=5, loss="hinge", max_iter=15000)
        svm_model.fit(tmp_X_train_Reco, tmp_Y_train_Reco)
        P_lr = svm_model.predict(tmp_X_test_Reco)
        cv_SVM_Reco.append(metrics.accuracy_score(tmp_Y_test_Reco, P_lr))


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



"""
#########################################
GAUSSIAN NAIVE BAYES
#########################################
"""
from sklearn.naive_bayes import MultinomialNB

cs=np.arange(0.001,0.5,0.005)

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






from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
clf = ExtraTreesClassifier(n_estimators=100, random_state=0,verbose=1)
clf.fit(X_train, Y_train)
y_pred_ert=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred_ert))

y_test=np.array(Y_test).flatten()
y_pred_ert=np.array(y_pred_ert).flatten()

"""print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
"""
plt.figure()
#labels = ['Base', 'Anes','Reco']
labels = ['Reco','Acute']
cm = confusion_matrix(Y_test, Y_pred_svm)
sn.heatmap(cm,annot=True,xticklabels=labels,yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')



'''
#################
K nearest Neighbors
################
'''
cs=np.arange(2,12)
knn_accuracy=[]
from sklearn.neighbors import KNeighborsClassifier

for c in cs:
    neigh = KNeighborsClassifier(n_neighbors=c)
    neigh.fit(X_train, Y_train)
    knn_predict = neigh.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, knn_predict)
    knn_accuracy.append(accuracy)
    print(c)

plt.plot(cs,knn_accuracy,color="blue")
plt.ylabel('KNN Accuracy [%]')
plt.xlabel('N')
plt.show()
#58% with 8nn




for r in range(0,2):
    for c in range (0,2):
        print(str(r)+str(c))
        X_test=X[(Y_ID == Part_reco[r]) | (Y_ID == Part_chro[c])]
        X_train=X[(Y_ID != Part_reco[r]) & (Y_ID != Part_chro[c])]
        Y_test=Y_out[(Y_ID == Part_reco[r]) | (Y_ID == Part_chro[c])]
        Y_train=Y_out[(Y_ID != Part_reco[r]) & (Y_ID != Part_chro[c])]


'''
#################
LASSO REGRESSION
################
'''
from sklearn import linear_model
clf = linear_model.Lasso()
clf.fit(X_train,Y_train)
Y_pred_lasso=clf.predict(X_test)

plt.plot(Y_pred_lasso)
np.where(Y_pred_lasso<0)


reg = linear_model.LassoCV(max_iter=100000)
reg.fit(X_train,Y_train)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X_test,Y_test))
coef = pd.Series(reg.coef_)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (8.0, 12.0)

matplotlib.rcParams.update({'font.size': 12})
imp_coef[0:30].plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

eoi=imp_coef.iloc[0:30]
eoe=eoi.index.values

imp_coef[-20:].plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

matrix = np.zeros((105,105))
#nr = np.arange(len(coef),0,-1)
nr = np.arange(0,len(coef))

a=0
for i in range (1,105):
    rng = np.arange(105 - i)
    fill=nr[a:a+len(rng)]
    matrix[rng, rng+i] = fill
    a=a+len(fill)

mat=pd.DataFrame(matrix)
mat = mat.applymap(str)
np.savetxt('foo.txt', mat, fmt='%s',delimiter=';')


plt.imshow(matrix)


i=0
e=1

for e in range(1,105 ):
    fill=nr[i:i + e]
    matrix[e, 105-e] = np.sqrt(fill + 1)
    i=i+e


    for i in range(1,105):
        matrix[105-i,105-e]=100

    np.fill_diagonal((matrix[:, 1:]),)
    matrix[np.diag_indices(104)]=100
    nr[i:i+e]
