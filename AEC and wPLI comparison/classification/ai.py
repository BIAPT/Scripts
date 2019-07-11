# Data manipulation
import scipy.io
import numpy as np
from data_manipulation import man


# Machine Learning 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import permutation_test_score
from sklearn.utils import resample

def classify(dataset, clf):
    # Initialize the Result data structures
    result = man.Result(dataset.technique, dataset.labels)

    # TODO: Check in the MATLAB file how to not have to do this + 1
    for test_id in range(1,dataset.num_participant+1):
        print("Participant: " + str(test_id) + " in hold-out set:")
        
        # Split the data in a leave one subject out manner
        (training_id, training_I) = dataset.prepare_training_test(test_id)

        # TODO: Missing hyperparameters tuning

        # This is where we test the classifier and where we set our results
    

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
def generate_confidence_interval(dataset, clf, num_bootstrap):

    # Here we overwrite the dataset X, y and I and run the classify function
    # for each bootstrap samples
    for id in range(num_bootstrap):
        print("Bootstrap sample #" + str(id))

        # Get the sampled with replacement dataset
        sample_X, sample_y, sample_I = resample(dataset.X, dataset.y, dataset.I)

        # Overiding dataset
        dataset.X = sample_X 
        dataset.y = sample_y
        dataset.I = sample_I 

        # Classify and get the results
        result = classify(dataset, clf)

        # TODO: append the result together

        result.summarize()
        