# General
import random

# Data manipulation
import scipy.io
import numpy as np

# Visualization
import matplotlib.pyplot as plt

def make_bar_plot(data1,conf1,data2,conf2,title):

    # Static data
    classes = ['Unconscious', 'Pre-ROC']

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
    ax.set_ylim([0,1])
    ax.legend()

    fig.tight_layout()
    plt.show()


# data for Linear SVM with C=0.5
mean_acc_wpli = [0.7903, 0.8322]
conf_acc_wpli = [(0.815 - 0.737)/2, (0.8241-0.7515)/2]
mean_acc_aec = [0.8922, 0.7273]
conf_acc_aec = [(0.911-0.854)/2, (0.7535 - 0.6691)/2]
make_bar_plot(mean_acc_wpli, conf_acc_wpli, mean_acc_aec, conf_acc_aec, 'Mean Accuracy Against Baseline Decoding Graph Theory')

# Unconsciousness : P  = 0.002
# pre-ROC: TODO