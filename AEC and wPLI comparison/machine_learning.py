# General
import random
import time

# Data manipulation
import scipy.io
import numpy as np

# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier

# Visualization
import matplotlib.pyplot as plt
from plot import viz



def make_mask(I,target):
    test_mask = np.where(I == target)
    mask = np.ones(len(I), np.bool)
    mask[test_mask] = 0
    train_mask = np.where(mask == 1)
    return (train_mask[0],test_mask[0])

def get_participant_data(I,X,y,p_id):
    rest_index, participant_index = make_mask(I,p_id)
    X_p = X[participant_index]
    y_p = y[participant_index]
    return (X_p,y_p)

def argmax(prob):
    guesses = []
    probabilities = []
    for guess in prob:
        probabilities.append(max(guess))
        guesses.append(np.argmax(guess))
    return (guesses,probabilities)

def get_best_estimate(y_pli,prob_pli,y_aec,prob_aec):
    y_pred = []
    amount_pli = 0
    amount_aec = 0
    for i in range(0,len(y_pli)):
        if(prob_pli[i] > prob_aec[i]):
            y_pred.append(y_pli[i])
            amount_pli += 1
        else:
            y_pred.append(y_aec[i])
            amount_aec += 1
    print("PLI: " + str(amount_pli) + " and AEC: " + str(amount_aec))
    return y_pred

def ensemble_classification(p_id, I, y, X_pli, X_aec, C, labels):
    print("Participant: " + str(p_id) + '--------------------')
    # Parameters
    target = [1,2,3,4,5,6,7,8,9]

    # Variables setup
    scores = list()
    cross_val_index = 0
    best_accuracy = -1

    best_accuracy_pli = -1
    best_accuracy_aec = -1
    best_accuracy_both = -1
    best_c_pli = 0
    best_c_aec = 0
    best_c_both = 0

    # Separate between training and test split
    train_mask, test_mask = make_mask(I, p_id)
    target.remove(p_id)

    I = np.delete(I, test_mask)

    X_train_pli, X_train_aec = X_pli[train_mask], X_aec[train_mask]
    X_test_pli, X_test_aec = X_pli[test_mask], X_aec[test_mask]

    y_test, y_train = y[test_mask], y[train_mask]


    # Split the training and test set and cross validate
    for target_i in target:
        print('Cross validation: ' + str(cross_val_index))

        # Get the splits
        train_index, validation_index = make_mask(I,target_i)

        # Get the training data
        training_X_pli = X_train_pli[train_index]
        training_X_aec = X_train_aec[train_index]
        training_y = y_train[train_index]

        # Get the validation data
        X_validation_pli = X_train_pli[validation_index]
        X_validation_aec = X_train_aec[validation_index]
        y_validation = y_train[validation_index]



        # train a one vs rest SVM
        # Create the model of SVM
        linear_clf_pli = svm.SVC(C=C[cross_val_index],kernel='linear',probability=True, verbose=False)
        linear_clf_aec = svm.SVC(C=C[cross_val_index],kernel='linear',probability=True, verbose=False)
        linear_clf_both = svm.SVC(C=C[cross_val_index],kernel='linear',probability=True, verbose=False)

        print("Training PLI Classifier...")
        linear_clf_pli.fit(training_X_pli, training_y)
        print("Training AEC Classifier...")
        linear_clf_aec.fit(training_X_aec, training_y)
        print("Training BOTH Classifier...")
        training_X_both = np.concatenate((training_X_pli,training_X_aec), axis = 1)
        linear_clf_both.fit(training_X_both, training_y)

        # Testing the classifier accuracy
        y_pred_pli = linear_clf_pli.predict(X_validation_pli)
        y_pred_aec = linear_clf_aec.predict(X_validation_aec)

        X_validation_both = np.concatenate((X_validation_pli,X_validation_aec), axis = 1)
        y_pred_both = linear_clf_both.predict(X_validation_both)


        current_acc_pli = accuracy_score(y_validation, y_pred_pli)
        current_acc_aec = accuracy_score(y_validation, y_pred_aec)
        current_acc_both = accuracy_score(y_validation, y_pred_both)

        if(current_acc_pli > best_accuracy_pli):
            best_c_pli = C[cross_val_index]
            best_accuracy_pli = current_acc_pli
            print("Best accuracy so far for PLI!")

        if(current_acc_aec > best_accuracy_aec):
            best_c_aec = C[cross_val_index]
            best_accuracy_aec = current_acc_aec
            print("Best accuracy so far for aec!")

        if(current_acc_both > best_accuracy_both):
            best_c_both = C[cross_val_index]
            best_accuracy_both = current_acc_both
            print("Best accuracy so far for both!")

        print("PLI: " + str(current_acc_pli))
        print("AEC: " + str(current_acc_aec))
        print("BOTH: " + str(current_acc_both))
        cross_val_index += 1

    linear_clf_pli = svm.SVC(C=best_c_pli,kernel='linear',probability=True, verbose=False)
    linear_clf_aec = svm.SVC(C=best_c_aec,kernel='linear',probability=True, verbose=False)
    linear_clf_both = svm.SVC(C=best_c_both,kernel='linear',probability=True, verbose=False)

    print("Training PLI Classifier...")
    linear_clf_pli.fit(X_train_pli, y_train)
    print("Training AEC Classifier...")
    linear_clf_aec.fit(X_train_aec, y_train)
    print("Training BOTH Classifier...")
    X_train_both = np.concatenate((X_train_pli,X_train_aec), axis = 1)
    linear_clf_both.fit(X_train_both, y_train)

    # Generalization for PLI
    y_pred_pli = linear_clf_pli.predict(X_test_pli)
    gen_acc_pli = accuracy_score(y_test, y_pred_pli)
    cm_pli = confusion_matrix(y_test, y_pred_pli)
    print("Generalization accuracy PLI: " + str(gen_acc_pli))

    # Generalization for AEC
    y_pred_aec = linear_clf_aec.predict(X_test_aec)
    gen_acc_aec = accuracy_score(y_test, y_pred_aec)
    cm_aec = confusion_matrix(y_test, y_pred_aec)
    print("Generalization accuracy AEC: " + str(gen_acc_aec))

    # Generalization for BOTH
    X_test_both = np.concatenate((X_test_pli,X_test_aec), axis = 1)
    y_pred_both = linear_clf_both.predict(X_test_both)
    gen_acc_both = accuracy_score(y_test, y_pred_both)
    cm_both = confusion_matrix(y_test, y_pred_both)
    print("Generalization accuracy BOTH: " + str(gen_acc_both))

    prob_pli = linear_clf_pli.predict_proba(X_test_pli)
    prob_pli = np.max(prob_pli, axis=1)
    prob_aec = linear_clf_aec.predict_proba(X_test_aec)
    prob_aec = np.max(prob_aec, axis=1)
    y_ens_prob = get_best_estimate(y_pred_pli,prob_pli,y_pred_aec,prob_aec)
    gen_acc_ensemble = accuracy_score(y_test, y_ens_prob)
    cm_ensemble = confusion_matrix(y_test, y_ens_prob)

    print("Generalization accuracy ENSEMBLE: " + str(gen_acc_ensemble))
    print('-----------------------------')

    return (gen_acc_pli,gen_acc_aec,gen_acc_both,gen_acc_ensemble,cm_pli,cm_aec,cm_both,cm_ensemble)

def split_data(X,y,split=0.1,seed=4):
    # Randomize the dataset for the split

    dataset = list(zip(X,y))
    random.Random(seed).shuffle(dataset)

    X = []
    y = []
    for element in dataset:
        X_element, y_element = element
        X.append(X_element)
        y.append(y_element)

    size_test = int(len(X)*split)

    X_test = X[:size_test]
    y_test = y[:size_test]
    X_train = X[size_test:]
    y_train = y[size_test:]

    return (X_train,X_test, y_train, y_test)

def load_data():
    data = scipy.io.loadmat('data/X_aec.mat')
    X_aec = np.array(data['X'])
    data = scipy.io.loadmat('data/X_pli.mat')
    X_pli = np.array(data['X'])
    data = scipy.io.loadmat('data/Y.mat')
    y = np.array(data['Y'])
    y = y[0]
    data = scipy.io.loadmat('data/I.mat');
    I = np.array(data['I']);
    I = I[0]
    return (X_pli,X_aec,y,I)




# Load X and Y data for classification
# X = (number sample, number features)
# Y = (number sample)
X_pli, X_aec, y, I = load_data()
print(X_pli.shape)
print(X_aec.shape)
X_both = np.concatenate((X_pli,X_aec), axis = 1)
print(X_both.shape)

# Iterate over the participant
C = [0.01,0.1,1,10,20,30,40,50]
labels = ['Baseline','Recovery']

accuracies_pli = []
cm_total_pli = []

accuracies_aec = []
cm_total_aec = []

accuracies_both = []
cm_total_both = []

accuracies_ensemble = []
cm_total_ensemble = []
for p_id in range(1,10):

    gen_acc_pli,gen_acc_aec,gen_acc_both,gen_acc_ensemble,cm_pli,cm_aec,cm_both,cm_ensemble = ensemble_classification(p_id, I, y, X_pli, X_aec, C, labels)

    #gen_acc_both,cm_both = cross_validation(p_id, I, X_both , y, 0.1)
    accuracies_pli.append(gen_acc_pli)
    accuracies_aec.append(gen_acc_aec)
    accuracies_both.append(gen_acc_both)
    accuracies_ensemble.append(gen_acc_ensemble)

    if(p_id == 1):
        cm_total_pli = cm_pli
        cm_total_aec = cm_aec
        cm_total_both = cm_both
        cm_total_ensemble = cm_ensemble
    else:
        cm_total_pli = np.add(cm_total_pli,cm_pli)
        cm_total_aec = np.add(cm_total_aec,cm_aec)
        cm_total_both = np.add(cm_total_both,cm_both)
        cm_total_ensemble = np.add(cm_total_ensemble,cm_ensemble)

print(accuracies_pli)
print("Generalization PLI: " + str(np.mean(accuracies_pli)))
viz.plot_confusion_matrix(cm_total_pli,labels,normalize=True)
plt.show()

print(accuracies_aec)
print("Generalization AEC: " + str(np.mean(accuracies_aec)))
viz.plot_confusion_matrix(cm_total_aec,labels,normalize=True)
plt.show()


print(accuracies_both)
print("Generalization BOTH: " + str(np.mean(accuracies_both)))
viz.plot_confusion_matrix(cm_total_both,labels,normalize=True)
plt.show()


print(accuracies_ensemble)
print("Generalization ENSEMBLE: " + str(np.mean(accuracies_ensemble)))
viz.plot_confusion_matrix(cm_total_ensemble,labels,normalize=True)
plt.show()
