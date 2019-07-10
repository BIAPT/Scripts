# Data manipulation
import scipy.io
import numpy as np
from data_manipulation import man

# Machine Learning 
from sklearn import svm
from sklearn import discriminant_analysis

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
        if classifier_type == 'lda':
            clf = discriminant_analysis.LinearDiscriminantAnalysis(solver='svd')
        elif classifier_type == 'linear svm':
            clf = svm.SVC(kernel='linear', verbose=False)
        elif classifier_type == 'rbf svm':
            clf = svm.SVC(kernel='rbf', verbose=False)
        elif classifier_type == 'poly svm':
            clf = svm.SVC(kernel='poly', verbose=False)
        else:
            exit("Classifier type not supported")

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
