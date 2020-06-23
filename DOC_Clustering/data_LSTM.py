from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import torch
torch.manual_seed(1)
import sys
sys.path.append('../')

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = [torch.tensor(ll, dtype=torch.long) for ll in labels]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def prepare_data_LSTM(data,Part_chro,Part_reco,stepsize_c,stepsize_r,windowsize,):
    s=0
    labels = []
    ID = []
    data_sample = []

    for p in Part_chro:
        tmp = data.query("ID == '{}'".format(p))
        for i in range(math.floor(len(tmp)/stepsize_c)):
            tmp2 = tmp.iloc[i*stepsize_c:i*stepsize_c+windowsize, :]
            tmp2.insert(0, 'sample', s)
            data_sample.append(tmp2)
            s += 1
            labels.append('c')
            ID.append(p)

    for p in Part_reco:
        tmp = data.query("ID == '{}'".format(p))
        for i in range(math.floor(len(tmp)/stepsize_r)-1):
            tmp2 = tmp.iloc[i*stepsize_r:i*stepsize_r+windowsize, :]
            tmp2.insert(0, 'sample', s)
            data_sample.append(tmp2)
            s += 1
            labels.append('r')
            ID.append(p)

    data_sample=pd.concat(data_sample)
    ID = pd.DataFrame(ID)

    labels = pd.DataFrame(labels).astype('category')
    encode_map = {'r': 1, 'c': 0}
    labels.replace(encode_map, inplace=True)
    labels = labels.values.tolist()

    areas = ['FC', 'FP', 'FO', 'FT', 'TO', 'TC', 'TP', 'PO', 'PC', 'CO', 'FF', 'CC', 'PP', 'TT', 'OO']

    data_by_sample = list(data_sample.groupby('sample'))
    data_by_sample = [dd[1][areas].values for dd in data_by_sample]

    dataset = MyDataset(data_by_sample, labels)
    datasize=int(len(dataset))

    return dataset, ID


