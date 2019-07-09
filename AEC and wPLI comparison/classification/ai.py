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

def classify(dataset, classifier_type):
    # Initialize the Result data structures
    result = man.Result(dataset.technique, dataset.labels)

    # TODO: Check in the MATLAB file how to not have to do this + 1
    for test_id in range(1,dataset.num_participant+1):
        print("Participant: " + str(test_id) + " in hold-out set:")
        
        # Split the data in a leave one subject out manner
        (training_id, training_I) = dataset.prepare_training_test(test_id)

        # TODO: Missing hyperparameters tuning

        # This is where we test the classifier and where we set our results
        
        # Creating our model
        if classifier_type == 'linear svm':
            clf = svm.SVC(kernel='linear', verbose=True)
        elif classifier_type == 'rbf svm':
            clf = svm.SVC(kernel='rbf', verbose=True)
        elif classifier_type == 'poly svm':
            clf = svm.SVC(kernel='poly', verbose=True)
        else:
            exit("Classifier type not supported")
        # Fitting our model
        clf.fit(dataset.X_train, dataset.y_train)

        # predicting
        y_pred = clf.predict(dataset.X_test)
        acc = accuracy_score(dataset.y_test, y_pred) # This might need to change we'll see
        cm = confusion_matrix(dataset.y_test, y_pred)
        
        print("Generalization accuracy: " + str(acc))

        # Saving the results
        result.add_acc(acc)
        result.add_cm(cm)

    return result
