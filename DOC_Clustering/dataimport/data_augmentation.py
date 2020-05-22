import matplotlib
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import metrics
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import random
import sys
sys.path.append('../')
import dataimport.prepareDataset as prep


[X_Base_d,X_Anes_d,X_Reco_d,Y_ID_Base_d,Y_ID_Anes_d,Y_ID_Reco_d,Y_out_Anes_d,Y_out_Base_d,Y_out_Reco_d ]=prep.prepare_Dataset('data/NEW_dPLI_all_10_1_left.pickle')

[X_Base_w,X_Anes_w,X_Reco_w,Y_ID_Base_w,Y_ID_Anes_w,Y_ID_Reco_w,Y_out_Anes_w,Y_out_Base_w,Y_out_Reco_w ]=prep.prepare_Dataset('data/NEW_wPLI_all_10_1_left.pickle')


datafile='data/NEW_dPLI_all_10_1_left.pickle'

[X,Y_out,Y_ID]=prep.prepare_Contrast_Dataset('data/contrast_NEW_dPLI_all_10_1_left.pickle')
