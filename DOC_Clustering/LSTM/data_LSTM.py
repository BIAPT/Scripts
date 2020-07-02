from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import torch
torch.manual_seed(1)
import sys
sys.path.append('../')
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = [torch.tensor(ll, dtype=torch.long) for ll in labels]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def prepare_data_LSTM(data,Part_chro,Part_reco,stepsize_c,stepsize_r,windowsize,Phase):
    s=0
    labels = []
    ID = []
    data_sample = []

    for p in Part_chro:
        # filter one participant
        if data.query("ID == '{}'".format(p)).shape[0]>0:
            tmp = data.query("ID == '{}'".format(p))
            i = 0

            # select Baseline and Anesthesia
            if Phase=='combined':
                tmp_p1 = tmp.query("Phase == '{}'".format('Base'))
                tmp_p2 = tmp.query("Phase == '{}'".format('Anes'))
            else:
                # select one phase
                tmp_p1 = tmp.query("Phase == '{}'".format(Phase))

            while True:
                # Selected Phase_ cut window of interest
                tmp2_p1 = tmp_p1.iloc[i*stepsize_c:i*stepsize_c+windowsize, :]
                tmp2_p1.insert(0, 'sample', s)

                if Phase=='combined':
                    # do the same with the Anesthesia data
                    tmp2_p2 = tmp_p2.iloc[i * stepsize_c:i * stepsize_c + windowsize, :]
                    tmp2_p2.insert(0, 'sample', s)
                    # combine Baseline and Anesthesia in the time dimension
                    tmp2_p1=pd.DataFrame(np.row_stack([tmp2_p1,tmp2_p2]))

                    if tmp2_p1.shape[0] < 2*windowsize:
                        # stop if time series is too short
                        break

                if tmp2_p1.shape[0]<windowsize:
                    break

                data_sample.append(tmp2_p1)
                s += 1
                labels.append('c')
                ID.append(p)
                i += 1

    # do the same thing for recovered patients
    for p in Part_reco:
        if data.query("ID == '{}'".format(p)).shape[0] > 0:
            # filter one participant
            tmp = data.query("ID == '{}'".format(p))
            i = 0

            # select Baseline and Anesthesia
            if Phase=='combined':
                tmp_p1 = tmp.query("Phase == '{}'".format('Base'))
                tmp_p2 = tmp.query("Phase == '{}'".format('Anes'))
            else:
                # select one phase
                tmp_p1 = tmp.query("Phase == '{}'".format(Phase))

            while True:
                # Selected Phase_ cut window of interest
                tmp2_p1 = tmp_p1.iloc[i * stepsize_r:i * stepsize_r + windowsize, :]
                tmp2_p1.insert(0, 'sample', s)

                if Phase=='combined':
                    # do the same with the Anesthesia data
                    tmp2_p2 = tmp_p2.iloc[i * stepsize_r:i * stepsize_r + windowsize, :]
                    tmp2_p2.insert(0, 'sample', s)
                    # combine Baseline and Anesthesia in the time dimension
                    tmp2_p1 = pd.DataFrame(np.row_stack([tmp2_p1, tmp2_p2]))

                    if tmp2_p1.shape[0] < 2 * windowsize:
                        # stop if time series is too short
                        break

                if tmp2_p1.shape[0] < windowsize:
                    break

                data_sample.append(tmp2_p1)
                s += 1
                labels.append('r')
                ID.append(p)
                i +=1

    data_sample=pd.concat(data_sample)
    data_sample.columns=data.columns.insert(0,'sample')
    ID = pd.DataFrame(ID)

    labels = pd.DataFrame(labels).astype('category')
    encode_map = {'r': 1, 'c': 0}
    labels.replace(encode_map, inplace=True)
    labels = labels.values.tolist()

    areas = ['FC', 'FP', 'FO', 'FT', 'TO', 'TC', 'TP', 'PO', 'PC', 'CO', 'FF', 'CC', 'PP', 'TT', 'OO']

    data_by_sample = list(data_sample.groupby('sample'))
    data_by_sample = [dd[1][areas].values for dd in data_by_sample]

    dataset = MyDataset(data_by_sample, labels)

    return dataset, ID

