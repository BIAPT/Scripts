import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import matplotlib.backends.backend_pdf
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from hmmlearn import hmm
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from clustering import markovchain
import numpy as np


healthy_data=pd.read_pickle('data/HEALTHY_Part_WholeBrain_wPLI_10_1_alpha.pickle')
doc_data=pd.read_pickle('data/New_Part_WholeBrain_wPLI_10_1_alpha.pickle')

data=pd.DataFrame(np.row_stack((doc_data,healthy_data)))
data.columns=healthy_data.columns

Phase=['Base']

Part = ['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15','S16','S17',
        'S18', 'S19', 'S20', 'S22', 'S23',
        'W03', 'W04', 'W08', 'W22', 'W28','W31', 'W34', 'W36',
        'A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']
Part_heal = ['A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']
Part_nonr = ['S05', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
             'S18', 'S22', 'S23', 'W04', 'W08', 'W28', 'W31', 'W34', 'W36']
Part_reco=['S02', 'S07', 'S09', 'S19', 'S20', 'W03', 'W22']

data_phase=data.query("Phase=='Base'")
X=data_phase.iloc[:,4:]

# Assign outcome
Y_out=np.zeros(len(X))
Y_out[data_phase['ID'].isin(Part_reco)] = 1
Y_out[data_phase['ID'].isin(Part_heal)] = 3

# create HMM
rremodel = hmm.GaussianHMM(n_components=5, covariance_type="full", n_iter=100)
rremodel.fit(X)
rremodel.monitor_.converged

Z2 = rremodel.predict(X)
rremodel.score(X)

P = rremodel.transmat_


P = np.array([[0.8, 0.2,0.1,0.1,0.1],[0.8, 0.2,0.1,0.1,0.1],[0.8, 0.2,0.1,0.1,0.1], [0.1, 0.9,0.1,0.1,0.1], [0.1, 0.9,0.1,0.1,0.1]]) # Transition matrix
mc = markovchain.MarkovChain(P, ['1', '2','3','4','5'])
mc.draw("markov-chain-two-states.png")


