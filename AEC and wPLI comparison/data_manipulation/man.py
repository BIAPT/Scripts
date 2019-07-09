# General
import random

# Data manipulation
import scipy.io
import numpy as np
import pickle

# Visualization
from ../plot import viz

# Classes
class Dataset:
    def __init__(self, technique, C, labels, num_participant, saving_path="/data/result.pickle"):
        # Load the data
        X_pli, X_aec, y, I = load_data()
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
        self.saving_path = saving_path

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
        train_mask, test_mask = make_mask(self.I, test_id)

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

    def save(self):
        pickle_out = open(self.saving_path,"wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def load(loading_path):
        pickle_in = open(loading_path,"rb")
        dataset = pickle.load(pickle_in)
        return dataset

# Helper data structure to hold information about results and
# print out a summary of the result
class Result:
    def __init__(self,technique,labels,saving_path="/data/result.pickle"):
        # Set default
        self.accuracies = []
        self.cm_total = []
        # Set the data
        self.technique = technique
        self.labels = labels
        self.saving_path = saving_path

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

    def save(self):
        pickle_out = open(self.saving_path,"wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def load(loading_path):
        pickle_in = open(loading_path,"rb")
        dataset = pickle.load(pickle_in)
        return dataset

# Make a mask to separate training from test participant
def make_mask(I,target):
    test_mask = np.where(I == target)
    mask = np.ones(len(I), np.bool)
    mask[test_mask] = 0
    train_mask = np.where(mask == 1)
    return (train_mask[0],test_mask[0])

# Loader function to get the data out from the .mat structure and into numpy array
# data are located /data folder
def load_data():
    data = scipy.io.loadmat('data/X_aec.mat')
    X_aec = np.array(data['X'])
    data = scipy.io.loadmat('data/X_pli.mat')
    X_pli = np.array(data['X'])
    data = scipy.io.loadmat('data/Y.mat')
    y = np.array(data['Y'])
    y = y[0]
    data = scipy.io.loadmat('data/I.mat')
    I = np.array(data['I'])
    I = I[0]
    return (X_pli,X_aec,y,I)
