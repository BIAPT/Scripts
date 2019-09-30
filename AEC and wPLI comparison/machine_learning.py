'''
Yacine Mahdid 2019-07-09 (yyyy/mm/dd)
This script was created in the context of the comparison of source localized aec and wpli performance in classifying various conscious state in healthy subject.
This python script purpose is to run the classification, the preprocessing is currently done in the MATLAB script name 'preprocess.m'
This script make use of sklearn, scipy, numpy, matplotlib and pickle. The import are being carried on in their relative module. 
'''

# Data manipulation
from data_manipulation import man
import pickle

# Machine Learning 
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from classification import ai

# Visualization
from plot import viz

# Initialize the variables
labels = ['Baseline','Induction']
num_participant = 9
num_permutation = 1000
num_bootstrap = 5000
technique = "wPLI"
clfs = [LinearDiscriminantAnalysis(solver='svd'), SVC(kernel='linear', C=0.1),SVC(kernel='linear', C=0.5), SVC(kernel='linear', C=1.0), SVC(kernel='rbf', C=0.1), SVC(kernel='rbf',C=1.0)]

selected_classifier = SVC(kernel='linear', C=0.50)

# Create the dataset
dataset = man.Dataset(technique, labels, num_participant)

# Classify the dataset and gather the result
result = ai.classify(dataset, selected_classifier, 2)

# Summarize the result
result.summarize(print_figure=False)

# Get full weights
#weights_mean, weights_std = ai.get_weight(dataset, selected_classifier, 3)

#print(len(weights_mean))
#print(len(weights_std))
#print(weights_mean)
#print(weights_std)

#man.save_data(weights_mean, weights_std, 'aec_pre_roc.mat')
# Do permutation testing on the chosen classifier
#(accuracy, permutation_scores, p_value) = ai.permutation_test(dataset, selected_classifier, num_permutation)
#print("Accuracy: " + str(accuracy))
#print("All scores: " + str(permutation_scores))
#print("Best p_value: " + str(p_value))

# Generate confidence interval for the classifier
#(accuracies,conf_interval_accuracy, conf_interval_baseline_f1, conf_interval_other_f1) = ai.generate_confidence_interval(dataset, selected_classifier, num_bootstrap,3)

#print("Confidence interval for Accuracy: " + str(conf_interval_accuracy))
#print("Confidence interval for Baseline F1: " + str(conf_interval_baseline_f1))
#print("Confidence interval for Other F1: " + str(conf_interval_other_f1))

#pickle_out = open("data/accuracies_"+technique,"wb")
#pickle.dump(accuracies, pickle_out)
#pickle_out.close()
