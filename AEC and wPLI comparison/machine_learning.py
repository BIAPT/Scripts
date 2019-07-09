# Data manipulation
from data_manipulation import man

# Machine Learning 
from classification import ai

# Visualization
from plot import viz

# Initialize the variables
# Doesn't make much sense to try C higher than 1 as we have noisy data
C = [0.01,0.1,1,10,20,30,40,50] 
labels = ['Baseline','Recovery']
num_participant = 9
technique = "wPLI"

# Create the dataset
dataset = man.Dataset(technique, C, labels, num_participant)

# Classify the dataset and gather the result
result = ai.classify(dataset)

# Summarize the result
result.summarize()

# TODO: Make the brain visualization

# TODO: Pickle the result datastructure into the data folder
