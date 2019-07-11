# General
import random

# Data manipulation
import scipy.io
import numpy as np

# Visualization
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap= plt.cm.Blues, print_figure=False):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    if not print_figure:
        return    

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set the colorbar
    if normalize:
        im.set_clim(vmin=0, vmax=1)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.min()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] != thresh else "black")
    fig.tight_layout()
    return ax

def plot_random_vector(X,Y,target):
    y = -1
    while(y != target):
        index = random.randint(0,len(Y)-1)
        x = X[index]
        y = Y[index]
    x = np.reshape(x, (82, 82))
    plt.imshow(x, cmap='jet')
    plt.colorbar()
    plt.show()
    print(y)