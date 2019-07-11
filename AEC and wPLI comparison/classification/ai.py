# Data manipulation
import scipy.io
import numpy as np
from data_manipulation import man
from math import floor
import copy

# Machine Learning 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import permutation_test_score
from sklearn.utils import resample
from sklearn.base import clone


def classify(dataset, original_clf):
    clf = clone(original_clf)
    # Initialize the Result data structures
    result = man.Result(dataset.technique, dataset.labels)

    # TODO: Check in the MATLAB file how to not have to do this + 1
    for test_id in range(1,dataset.num_participant+1):
        print("Participant: " + str(test_id) + " in hold-out set:")
        
        # Split the data in a leave one subject out manner
        dataset.prepare_training_test(test_id)

        # Fitting our model
        clf.fit(dataset.X_train, dataset.y_train)

        # predicting
        y_pred = clf.predict(dataset.X_test)
        cm = confusion_matrix(dataset.y_test, y_pred)

        report = classification_report(dataset.y_test, y_pred, output_dict=True)
        
        print("Generalization accuracy: " + str(report['accuracy']))

        # Saving the results
        result.add_cm(cm)
        result.add_report(report)

    return result

# Will run the classification num_permutation times to create a distribution in order to get
# a p_value for the score value of the classifier clf
def permutation_test(dataset, clf, num_permutation):
    train_test_splits = man.generate_train_test_splits(dataset)
    (accuracy, permutation_scores, p_value) = permutation_test_score(clf, dataset.X, dataset.y, groups=dataset.I, cv=train_test_splits, n_permutations=num_permutation, verbose=num_permutation)
    return (accuracy, permutation_scores, p_value)

# Iterate num_bootstrap times and create a classifier with the resampled data
# Then create confidence interval for the and the accuracy, f_1 score
# The p value used here is 0.05
# This means that the lower bound = math.floor((num_bootstrap/100)*2.5)
#                     upper bound = math.floor((num_bootstrap/100)*97.5)
def generate_confidence_interval(original_dataset, clf, num_bootstrap):
    lb_index = floor((num_bootstrap/100)*(2.5))
    ub_index = floor((num_bootstrap/100)*(97.5))

    accuracies = []
    baseline_f1s = []
    other_f1s = []

    conf_interval_accuracy = (-1,-1)
    conf_interval_baseline_f1 = (-1,-1)
    conf_interval_other_f1 = (-1,-1)

    # Here we overwrite the dataset X, y and I and run the classify function
    # for each bootstrap samples
    for id in range(num_bootstrap):
        print("Bootstrap sample #" + str(id))
        # Copy the original dataset before manipulating it
        dataset = copy.deepcopy(original_dataset)

        # Get the sampled with replacement dataset
        sample_X, sample_y, sample_I = resample(dataset.X, dataset.y, dataset.I)

        # Overiding dataset
        dataset.X = sample_X 
        dataset.y = sample_y
        dataset.I = sample_I 

        dataset.reset_variables()

        # Classify and get the results
        result = classify(dataset, clf)

        accuracies.append(result.get_mean_acc())
        baseline_f1s.append(result.get_mean_baseline_f1())
        other_f1s.append(result.get_mean_other_f1())
        
        result.summarize()
    
    # Sort the results
    accuracies.sort()
    baseline_f1s.sort()
    other_f1s.sort()

    # Set the confidence interval at the right index
    conf_interval_accuracy = (accuracies[lb_index],accuracies[ub_index])
    conf_interval_baseline_f1 = (baseline_f1s[lb_index],baseline_f1s[ub_index])
    conf_interval_other_f1 = (other_f1s[lb_index],other_f1s[ub_index])

    return (conf_interval_accuracy, conf_interval_baseline_f1, conf_interval_other_f1)