'''
Yacine Mahdid 2019-07-09 (yyyy/mm/dd)
This script was created in the context of the comparison of source localized aec and wpli performance in classifying various conscious state in healthy subject.
This python script purpose is to run the classification, the preprocessing is currently done in the MATLAB script name 'preprocess.m'
This script make use of sklearn, scipy, numpy, matplotlib and pickle. The import are being carried on in their relative module. 
'''

# Data manipulation
from data_manipulation import man

# Machine Learning 
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from classification import ai

# Visualization
from plot import viz

# Initialize the variables
labels = ['Baseline','Recovery']
num_participant = 9
num_permutation = 20
num_bootstrap = 1000
technique = "wPLI"
clfs = [LinearDiscriminantAnalysis(solver='svd'), SVC(kernel='linear'), SVC(kernel='rbf'), SVC(kernel='poly')]

# Create the dataset
dataset = man.Dataset(technique, labels, num_participant)

# Classify the dataset and gather the result
#result = ai.classify(dataset, clfs[0])

# Do permutation testing on the chosen classifier
#(accuracy, permutation_scores,p_value) = ai.permutation_test(dataset, clfs[0], num_permutation)
#print("Accuracy: " + str(accuracy))
#print("All scores: " + str(permutation_scores))
#print("Best p_value: " + str(p_value))

# Generate confidence interval for the classifier
ai.generate_confidence_interval(dataset, clfs[0], num_bootstrap)

# Save the result and the dataset into the data folder
#result.save()
#dataset.save()

# Summarize the result
#result.summarize()
