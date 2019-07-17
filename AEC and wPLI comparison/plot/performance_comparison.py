# General
import random

# Data manipulation
import scipy.io
import numpy as np

# Visualization
import matplotlib.pyplot as plt

def make_bar_plot(data1,conf1,data2,conf2,title):

    # Static data
    classes = ['Induction', 'Unconscious', 'Pre-ROC', 'Recovery']

    ind = np.arange(len(data1))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, data1, width, label='wPLI', yerr=conf1)
    rects2 = ax.bar(ind + width/2, data2, width, label='AEC', yerr=conf2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(classes)
    ax.legend()

    fig.tight_layout()
    plt.show()

# data for Linear SVM with C=0.5
mean_acc_wpli = [0.5505, 0.7849, 0.8041, 0.6098]
conf_acc_wpli = [0, (0.8116 - 0.7354)/2, (0.8241-0.7515)/2, 0]
mean_acc_aec = [0.5394, 0.8641, 0.7276, 0.5602]
conf_acc_aec = [0, (0.8966 - 0.8367)/2, (0.7535 - 0.6691)/2, 0]
make_bar_plot(mean_acc_wpli, conf_acc_wpli, mean_acc_aec, conf_acc_aec, 'Mean Accuracy Against Baseline Decoding SVM')


# data for LDA with svg
mean_acc_wpli = [0.5206583420355466, 0.7930148717993689, 0.7809757871472132, 0.616840516323644]
conf_acc_wpli = [0, 0, 0, 0]
mean_acc_aec = [0.5247893150701697, 0.821774689973807, 0.6796940565270906, 0.5335461132479048]
conf_acc_aec = [0, 0, 0, 0]
make_bar_plot(mean_acc_wpli, conf_acc_wpli, mean_acc_aec, conf_acc_aec, 'Mean Accuracy Against Baseline Decoding LDA')