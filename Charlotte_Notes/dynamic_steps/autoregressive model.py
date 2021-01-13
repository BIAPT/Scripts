import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.backends.backend_pdf
from sklearn import preprocessing
from clustering import create_image

def build_time_delay_matrices(x, r):
    """
    Builds x1 and x2 for regression

    Args:
    x (numpy array of floats): data to be auto regressed
    r (scalar): order of Autoregression model

    Returns:
    (numpy array of floats) : to predict "x2"
    (numpy array of floats) : predictors of size [r,n-r], "x1"

    """
    # construct the time-delayed data matrices for order-r AR model
    x1 = np.ones(len(x)-r)
    x1 = np.vstack((x1, x[0:-r]))
    xprime = x
    for i in range(r-1):
        xprime = np.roll(xprime, -1)
        x1 = np.vstack((x1, xprime[0:-r]))

    x2 = x[r:]

    return x1, x2

def AR_model(x, r):
    """
    Solves Autoregression problem of order (r) for x

    Args:
    x (numpy array of floats): data to be auto regressed
    r (scalar): order of Autoregression model

    Returns:
    (numpy array of floats) : to predict "x2"
    (numpy array of floats) : predictors of size [r,n-r], "x1"
    (numpy array of floats): coefficients of length [r] for prediction after
    solving the regression problem "p"
    """
    x1, x2 = build_time_delay_matrices(x, r)

    # solve for an estimate of lambda as a linear regression problem
    p, res, rnk, s = np.linalg.lstsq(x1.T, x2, rcond=None)

    return x1, x2, p

def AR_prediction(x_test, p):
    """
    Returns the prediction for test data "x_test" with the regression
    coefficients p

    Args:
    x_test (numpy array of floats): test data to be predicted
    p (numpy array of floats): regression coefficients of size [r] after
    solving the autoregression (order r) problem on train data

    Returns:
    (numpy array of floats): Predictions for test data. +1 if positive and -1
    if negative.
    """
    x1, x2 = build_time_delay_matrices(x_test, len(p)-1)

    # Evaluating the AR_model function fit returns a number.
    # We take the sign (- or +) of this number as the model's guess.
    return (np.dot(x1.T, p))

def error_rate(x_test, p):
    """
    Returns the error of the Autoregression model. Error is the number of
    mismatched predictions divided by total number of test points.

    Args:
    x_test (numpy array of floats): data to be predicted
    p (numpy array of floats): regression coefficients of size [r] after
    solving the autoregression (order r) problem on train data

    Returns:
    (float): Error (percentage).
    """
    x1, x2 = build_time_delay_matrices(x_test, len(p)-1)

    return np.sum(abs(x2 - AR_prediction(x_test, p)))

def plot_residual_histogram(res):
  """Helper function for Exercise 4A"""
  fig = plt.figure()

  plt.hist(res)
  plt.xlabel('error in linear model')
  plt.title('stdev of errors = {std:.4f}'.format(std=res.std()))

  plt.show()

def plot_training_fit(x1, x2, p):
  """Helper function for Exercise 4B"""
  fig = plt.figure()

  plt.scatter(x2 + np.random.standard_normal(len(x2))*0.02,
              np.dot(x1.T, p), alpha=0.2)
  plt.title('Training fit, order {r:d} AR model'.format(r=r))

  plt.xlabel('x')
  plt.ylabel('estimated x')
  plt.show()

healthy_data=pd.read_pickle('data/HEALTHY_Part_WholeBrain_wPLI_10_1_alpha.pickle')
doc_data=pd.read_pickle('data/New_Part_WholeBrain_wPLI_10_1_alpha.pickle')
import seaborn as sns


data=pd.DataFrame(np.row_stack((doc_data,healthy_data)))
data.columns=healthy_data.columns

X=data.iloc[:,4:]
areas= X.columns

Phase=['Base']

Part = ['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15','S16','S17',
        'S18', 'S19', 'S20', 'S22', 'S23',
        'W03', 'W04', 'W08', 'W22', 'W28','W31', 'W34', 'W36',
        'A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']
Part_heal = ['A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']
Part_nonr = ['S05', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
             'S18', 'S22', 'S23', 'W04', 'W08', 'W28', 'W31', 'W34', 'W36']
Part_reco=['S02', 'S07', 'S09', 'S19', 'S20', 'W03', 'W22']


#monkey_at_typewriter = '10010101001101000111001010110001100101000101101001010010101010001101101001101000011110100011011010010011001101000011101001110000011111011101000011110000111101001010101000111100000011111000001010100110101001011010010100101101000110010001100011100011100011100010110010111000101'
#x = np.array([0,0,0,0,0,1,0,1,0,1])

x=np.array(X.loc[1:150 ,areas[0]], dtype='float')
test=np.array(X.loc[150:300 ,areas[0]], dtype='float')

r = 5 # remove later
x1, x2, p = AR_model(x, r)

with plt.xkcd():
  plot_training_fit(x1, x2, p)

# range of r's to try
r = np.arange(1, 21)
err = np.ones_like(r) * 1.0

for i, rr in enumerate(r):
  # fitting the model on training data
  x1, x2, p = AR_model(x, rr)
  # computing and storing the test error
  test_error = error_rate(test, p)
  err[i] = test_error

fig = plt.figure()
plt.plot(r, err, '.-')
#plt.plot([1, r[-1]], [0.5, 0.5], c='r', label='random chance')
plt.xlabel('Order r of AR model')
plt.ylabel('Test error')
plt.xticks(np.arange(0, 25, 5))
plt.legend()
plt.show()