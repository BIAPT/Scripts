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

# Classes
class Dataset:
    def __init__(self, technique, C, labels, num_participant):
        # Load the data
        X_pli, X_aec, y, I = man.load_data()
        X_both = np.concatenate((X_pli,X_aec), axis = 1)
        
        # Set dataset specific variable
        if technique == "wPLI":
            self.X = X_pli
        elif technique == "AEC":
            self.X = X_aec
        elif technique == "BOTH":
            self.X = X_both
        else:
            exit("Error with technique label!")

        self.y = y
        self.I = I
        # Set static variables for output
        self.C = C
        self.labels = labels
        self.num_participant = num_participant

        # Set working variables
        self.all_id = [id for id in range(1,num_participant+1)]
        # Set the training/validation/test instances
        self.X_train = []
        self.X_validation = []
        self.X_test = []
        
        self.y_train = []
        self.y_validation = []
        self.y_test = []

    def prepare_training_test(self, test_id):
        # Make the mask for the training and test dataset
        train_mask, test_mask = man.make_mask(self.I, test_id)

        # Setup the trianing and test dataset
        self.X_train = self.X[train_mask]
        self.X_test = self.X[test_mask]
        self.y_train = self.y[train_mask]
        self.y_test = self.y[test_mask]

        # Set the training ids
        training_id = self.all_id.copy()
        training_id.remove(test_id)

        # Create the training I
        training_I = self.I.copy()
        training_I = np.delete(self.training_I, test_mask)
        return (training_id, training_I)

    def prepare_training_validation(self, training_I, validation_id):
        # Get the splits
        train_index, validation_index = man.make_mask(self.training_I,validation_id)
        # TODO: FINISIH THIS PART

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
            self.cm_total.add(self.cm_total,cm)

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

def full_classification(p_id, I, y, X_pli, X_aec, C, labels):
    print("Participant: " + str(p_id) + '--------------------')

    # Parameters
    target = [1,2,3,4,5,6,7,8,9]

    # Variables setup
    cross_val_index = 0

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

def classify(dataset):

    # Initialize the Result data structures
    result = Result(dataset.technique, dataset.labels)

    # TODO: Check in the MATLAB file how to not have to do this + 1
    for test_id in range(1,dataset.num_participant+1):
        print("Participant: " + str(test_id) + " in hold-out set:")
        
        # Split the data in a leave one subject out manner
        (training_id, training_I) = dataset.prepare_training_test(test_id)

        for validation_id in training_id:
            print("This is where the hyperparameters optimization happens")
        
        # This is where we test the classifier and where we set our results
    
    return result

# Initialize the variables
# Doesn't make much sense to try C higher than 1 as we have noisy data
C = [0.01,0.1,1,10,20,30,40,50] 
labels = ['Baseline','Recovery']
num_participant = 9
technique = "wPLI"

# Create the dataset
dataset = Dataset(technique, C, labels, num_participant)

# Classify the dataset and gather the result
result = classify(dataset)

# Summarize the result
result.summarize()


# Iterate over each participant
for p_id in range(1,10):

    # TODO: Refactor this part to do the adding inside the ensemble function
    gen_acc_pli,gen_acc_aec,gen_acc_both,gen_acc_ensemble,cm_pli,cm_aec,cm_both,cm_ensemble = full_classification(p_id, I, y, X_pli, X_aec, C, labels)


    # Add the current result to what we already have
    result.add_acc(gen_acc_pli)

    result.add_cm(cm_pli)

# Summarize the data and plot it to the command line
result.summarize()
