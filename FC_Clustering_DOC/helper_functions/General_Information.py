import pandas as pd
import numpy as np


def get_data():
    # import data
    #healthy_data = pd.read_pickle('data/HEALTHY_Part_WholeBrain_wPLI_10_1_alpha.pickle')
    #doc_data = pd.read_pickle('data/New_Part_WholeBrain_wPLI_10_1_alpha.pickle')
    data = pd.read_pickle('data/33_Part_WholeBrain_wPLI_10_10_alpha.pickle')

    # combine both sets
    #data = pd.DataFrame(np.row_stack((doc_data, healthy_data)))
    #data.columns = healthy_data.columns
    data = data.query("Phase=='Base'")
    return data

AllPart= {}
# define groups of participants
AllPart["Part"] = ['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                    'S18', 'S19', 'S20', 'S22', 'S23',
                    'W03', 'W04', 'W08', 'W22', 'W28', 'W31', 'W34', 'W36',
                    'A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']

AllPart["Part_heal"] = ['A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']

AllPart["Part_nonr"] = ['S05', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17', 'S18', 'S22', 'S23', 'W04', 'W36']

AllPart["Part_ncmd"] = ['S19', 'W03', 'W08', 'W28', 'W31', 'W34']

AllPart["Part_reco"] = ['S02', 'S07', 'S09', 'S20',  'W22']

data = get_data()

# extract only the X- values
X = data.iloc[:, 4:]

# Assign outcome
Y_out = np.zeros(len(X))
Y_out[data['ID'].isin(AllPart["Part_ncmd"])] = 1
Y_out[data['ID'].isin(AllPart["Part_reco"])] = 2
Y_out[data['ID'].isin(AllPart["Part_heal"])] = 3

# number of Clusters/ Phases to explore
KS=[5, 6, 7, 8]





