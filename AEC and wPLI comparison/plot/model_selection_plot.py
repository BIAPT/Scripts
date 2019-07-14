# General
import random

# Data manipulation
import scipy.io
import numpy as np

# Visualization
import matplotlib.pyplot as plt

#Helper function
def make_plot(data,title):
    #static data
    models = ["LDA","Linear SVM C=0.1", "Linear SVM C=0.5", "Linear SVM C=1.0", "RBF SVM C=0.1", "RBF SVM C=1.0"]
    x = [0,1,2,3,4,5]

    # Figure 
    fig = plt.figure()
    ax = plt.axes()

    ax.plot(x, data)

    ax.set(xticks=np.arange(len(data) + 1),
        # ... and label them with the respective list entries
        xticklabels=models,
        title=title,
        ylabel='Accuracy',
        xlabel='Model Complexity')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

    plt.show()


#data
unconscious_wpli = [0.79301, 0.768681, 0.78495, 0.784681, 0.51537, 0.6866]
unconscious_aec = [0.82177, 0.87729, 0.8641, 0.86434, 0.8493, 0.8768]
pre_roc_wpli = [0.780975, 0.78513, 0.80415, 0.80397, 0.6099, 0.7107]
pre_roc_aec = [0.67969, 0.74109, 0.7276, 0.70856, 0.6378, 0.7579]

aggregate = [0.7688, 0.79304, 0.7952, 0.79036, 0.6530, 0.758]


make_plot(aggregate, "Mean Aggregate Accuracy Across Models")
