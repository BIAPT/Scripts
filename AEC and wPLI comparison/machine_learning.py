# General
import random
import time

# Data manipulation
import scipy.io
import numpy as np
from data_manipulation import man
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

# Helper function for the ensemble classification
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

# TODO: Rename this function to a better fitting name
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
    train_mask, test_mask = man.make_mask(I, p_id)
    target.remove(p_id)

    I = np.delete(I, test_mask)

    X_train_pli, X_train_aec = X_pli[train_mask], X_aec[train_mask]
    X_test_pli, X_test_aec = X_pli[test_mask], X_aec[test_mask]

    y_test, y_train = y[test_mask], y[train_mask]


    # Split the training and test set and cross validate
    for target_i in target:
        print('Cross validation: ' + str(cross_val_index))

        # Get the splits
        train_index, validation_index = man.make_mask(I,target_i)

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


# Classes
class Dataset:
    def __init__(self,C,labels):
        # Load the data from /data
        self.X_pli, self.X_aec, self.y, self.I = man.load_data()
        # Create the merged dataset
        self.X_both = np.concatenate((self.X_pli,self.X_aec), axis = 1)
        # Set the data
        self.C = C
        self.labels = labels

# Helper data structure to hold information about results and
# print out a summary of the result
class Result:
    def __init__(self,technique,labels):
        # Set default
        self.accuracies = []
        self.cm_total = []
        # Set the data
        self.technique = technique
        self.labels = labels
    
    def add_acc(self,accuracy):
        self.accuracies.append(accuracy)

    def add_cm(self,cm):
        if not self.cm_total:
            self.cm_total = cm
        else:
            self.cm_total.add(self.cm_stotal,cm)

    def print_acc(self):
        print("[" + str(self.technique) + "]")
        print(self.accuracies)
        print("Mean Accuracy: " + str(np.mean(self.accuracies)))

    def plot_cm(self):
        viz.plot_confusion_matrix(self.cm_total,self.labels,normalize=True)
        plt.show()
    
    def summarize(self):
        self.print_acc()
        self.plot_cm()


# TODO: Remove this
# Load X and Y data for classification
# X = (number sample, number features)
# Y = (number sample)
X_pli, X_aec, y, I = man.load_data()
X_both = np.concatenate((X_pli,X_aec), axis = 1)

# Initialize the variables
C = [0.01,0.1,1,10,20,30,40,50]
labels = ['Baseline','Recovery']

# Create the dataset
dataset = Dataset(C,labels)

# Initialize the Result data structures
result_pli = Result("wPLI",labels)
result_aec = Result("AEC",labels)
result_both = Result("Both",labels)
result_ensemble = Result("Ensemble",labels)

# TODO: Here should just need to call ensemble_classification and give all we have
# It should return the right stuff and we should just skip straight to the output part
# The for loop can easily be inside the classification

# Iterate over each participant
for p_id in range(1,10):

    # TODO: Refactor this part to do the adding inside the ensemble function
    gen_acc_pli,gen_acc_aec,gen_acc_both,gen_acc_ensemble,cm_pli,cm_aec,cm_both,cm_ensemble = ensemble_classification(p_id, I, y, X_pli, X_aec, C, labels)


    # Add the current result to what we already have
    result_pli.add_acc(gen_acc_pli)
    result_aec.add_acc(gen_acc_aec)
    result_both.add_acc(gen_acc_both)
    result_ensemble.add_acc(gen_acc_ensemble)

    result_pli.add_cm(cm_pli)
    result_aec.add_cm(cm_aec)
    result_both.add_cm(cm_both)
    result_ensemble.add_cm(cm_ensemble)

# Summarize the data and plot it to the command line
result_pli.summarize()
result_aec.summarize()
result_both.summarize()
result_ensemble.summarize()
