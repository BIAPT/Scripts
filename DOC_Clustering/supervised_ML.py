import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import random

random.seed(141)

data=pd.read_pickle('data/final_wPLI_clustering.pickle')
X=data.iloc[:,4:]
Y_ID=data.iloc[:,1]
Y_St=data.iloc[:,2]
Y_time=data.iloc[:,3]

X_train, X_test, Y_train, Y_test= train_test_split(X, Y_St, test_size=0.3, random_state=2)

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


'''
#################
SVM (sklearn) for plots
################
'''
cs=np.arange(0.3,3,0.1)
svm_accuracy=[]

for c in cs:
    model_svm = svm.LinearSVC(C = c, loss = "hinge", max_iter= 15000)
    model_svm.fit(X_train, Y_train)
    Y_pred_svm = model_svm.predict(X_test)
    svm_accuracy.append(metrics.accuracy_score(Y_test, Y_pred_svm))
    print(c)

plt.plot(cs,svm_accuracy)
plt.ylabel('SVM Accuracy [%]')
plt.xlabel('C')
plt.show()
#59 %


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


reg = linear_model.LassoCV()
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



'''
#################
Random Forest (sklearn)
################
'''
model_rf = RandomForestClassifier(verbose=1)
model_rf.fit(X_train, Y_train)
predictions_rf_2 = model_rf.predict(X_test)
metrics.accuracy_score(Y_test, predictions_rf_2)

'''
#################
Logistic Regression (sklearn)
################
'''
from sklearn.linear_model import LogisticRegression
#Fit Logistic Model
model_logistic = LogisticRegression()
model_logistic.fit(X_train, Y_train)
Y_pred_logistic = model_logistic.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred_logistic))


""""
#######################
    2nd approach
#######################
"""

#CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words = 'english')
vect_train = count_vect.fit_transform(X_train['review'])
vect_test = count_vect.transform(X_test['review'])

tf_idf_vectorizer = TfidfVectorizer(stop_words = 'english')
vect_train_idf = tf_idf_vectorizer.fit_transform(X_train)
vect_test_idf = tf_idf_vectorizer.transform(X_test)


'''
#################
Decision tree (sklearn)
################
'''
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, Y_train)
Y_pred_dt = model_dt.predict(X_test)
metrics.accuracy_score(Y_test, Y_pred_dt)


"""
#########################################
GAUSSIAN NAIVE BAYES
#########################################
"""
from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))



from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
clf = ExtraTreesClassifier(n_estimators=100, random_state=0,verbose=1)
clf.fit(X_train, Y_train)
y_pred_ert=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred_ert))




