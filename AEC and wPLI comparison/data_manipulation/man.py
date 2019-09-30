# General
import random

# Data manipulation
import scipy.io
import numpy as np
import pickle

# Visualization
import matplotlib.pyplot as plt
from plot import viz

# Classes
class Dataset:
    def __init__(self, technique, labels, num_participant, saving_path="./data/dataset.pickle"):
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
        self.technique = technique
        self.labels = labels
        self.num_participant = num_participant
        self.saving_path = saving_path

        # Set working variables
        self.all_id = [id for id in range(1,num_participant+1)]
        self.reset_variables()

    def reset_variables(self):
        # Set the training/validation/test instances
        self.train_mask = []
        self.test_mask = []
        
        self.X_train = []
        self.X_validation = []
        self.X_test = []
        
        self.y_train = []
        self.y_validation = []
        self.y_test = []

    def prepare_training_test(self, test_id):
        # Make the mask for the training and test dataset
        train_mask, test_mask = make_mask(self.I, test_id)

        self.train_mask = train_mask
        self.test_mask = test_mask

        # Setup the training and test dataset
        self.X_train = self.X[train_mask]
        self.X_test = self.X[test_mask]
        self.y_train = self.y[train_mask]
        self.y_test = self.y[test_mask]
        

    def prepare_training_validation(self, training_I, validation_id):
        # Get the splits
        train_mask, validation_mask = man.make_mask(training_I,validation_id)

        # Setup the training data
        self.X_train = self.X[self.train_mask]
        self.y_train = self.y[self.train_mask]

        # Get the validation data
        self.X_validation = self.X_train[validation_mask]
        self.y_validation = self.y_train[validation_mask]

        # Get the real training data
        self.X_train = self.X_train[train_mask]
        self.y_train = self.y_train[train_mask]

        # TODO: FINISIH THIS PART

    def save(self):
        pickle_out = open(self.saving_path,"wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def load(self, loading_path):
        pickle_in = open(loading_path,"rb")
        dataset = pickle.load(pickle_in)
        return dataset

# Helper data structure to hold information about results and
# print out a summary of the result
class Result:
    def __init__(self,technique,labels,saving_path="./data/result.pickle"):
        
        # Set default
        self.accuracies = []
        self.cm_total = []

        self.baseline_f1 = []
        self.other_f1 = []
        
        # Set the data
        self.technique = technique
        self.labels = labels
        self.saving_path = saving_path


    def add_cm(self,cm):
        if len(self.cm_total) == 0:
            self.cm_total = cm
        else:
            self.cm_total = np.add(self.cm_total,cm)

    def add_report(self, report, other_index):
        self.baseline_f1.append(report['0']['f1-score'])
        self.other_f1.append(report[str(other_index)]['f1-score'])
        #self.accuracies.append(report['accuracy'])

    def add_acc(self, accuracy):
        self.accuracies.append(accuracy)

    def get_mean_acc(self):
        return np.mean(self.accuracies)

    def get_mean_baseline_f1(self):
        return np.mean(self.baseline_f1)

    def get_mean_other_f1(self):
        return np.mean(self.other_f1)

    def print_acc(self):
        print("Mean Accuracy: " + str(self.get_mean_acc()))

    def print_f1(self):
        print("Mean F1 " + self.labels[0] + ": " + str(self.get_mean_baseline_f1()))
        print("Mean F1 " + self.labels[1] + ": " + str(self.get_mean_other_f1()))


    def plot_cm(self, print_figure=False):
        viz.plot_confusion_matrix(self.cm_total,self.labels,normalize=True, print_figure=print_figure)
        if print_figure:
            plt.show()

    def summarize(self, print_figure):
        print("Analysis Technique: " + str(self.technique))
        self.print_acc()
        self.print_f1()
        self.plot_cm(print_figure=print_figure)

    def save(self):
        pickle_out = open(self.saving_path,"wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def load(self, loading_path):
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

def save_data(mean,std, filename):
    data = {'weights_mean':mean,'weights_std':std}
    scipy.io.savemat(filename,data)

# Helper function to create a tuple list of train and test splits
def generate_train_test_splits(dataset):
    train_test_splits = []

    # Create the train test split for all LOSO participant
    for test_id in range(1,dataset.num_participant+1):

        # Split the data into a LOSO cross validated train/test split
        dataset.prepare_training_test(test_id)

        # Get the train and test split
        train_split = dataset.train_mask
        test_split = dataset.test_mask

        # Add them to the return list
        train_test_splits.append((train_split,test_split))

    return train_test_splits