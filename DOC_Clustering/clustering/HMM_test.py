import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
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

X_DOC=X.iloc[np.where(Y_out==1)]
X_HEA=X.iloc[np.where(Y_out==3)]
X_HEA.shape

# create HMM
rremodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
rremodel.fit(X_HEA[0:10])
HZH = rremodel.predict(X_HEA[0:10])
#HZD = rremodel.predict(X_DOC)
#rremodel.score(X)

rremodel.transmat_

mc = markovchain.MarkovChain(one_step_array, ['1', '2'])
mc = markovchain.MarkovChain(rremodel.transmat_ , ['1', '2'])
mc.draw("markov-chain-two-states.png")

transitions = HZH
n = 2

def onestep_transition_matrix(transitions,n_states):
    n = n_states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
            #row[:] = [f/len(transitions) for f in row]
    return M

np.array(onestep_transition_matrix(HZH,3))
np.array(v2onestep_transition_matrix(HZH,2))
rremodel.transmat_


sample = [1,1,2,2,1,3,2,1,2,3,1,2,3,1,2,3,1,2,1,2]
