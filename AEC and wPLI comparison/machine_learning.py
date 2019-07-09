'''
Yacine Mahdid 2019-07-09 (yyyy/mm/dd)
This script was created in the context of the comparison of source localized aec and wpli performance in classifying various conscious state in healthy subject.
This python script purpose is to run the classification, the preprocessing is currently done in the MATLAB script name 'preprocess.m'
This script make use of sklearn, scipy, numpy, matplotlib and pickle. The import are being carried on in their relative module. 
'''

# Data manipulation
from data_manipulation import man

# Machine Learning 
from classification import ai

# Visualization
from plot import viz

# Initialize the variables
C = [0.001, 0.05, 0.1, 0.15, 0.20, 0.25, 0.5, 0.75, 1, 2] 
labels = ['Baseline','Recovery']
num_participant = 9
technique = "wPLI"
classifier_types = ['linear svm', 'rbf svm', 'poly svm']
classifier_type = 'rbf svm'

# Create the dataset
dataset = man.Dataset(technique, C, labels, num_participant)

# Classify the dataset and gather the result
result = ai.classify(dataset, classifier_type)

# Save the result and the dataset into the data folder
#result.save()
#dataset.save()

# Summarize the result
result.summarize()
