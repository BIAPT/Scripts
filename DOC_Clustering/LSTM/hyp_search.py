import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import torch
torch.manual_seed(1)
import sys
sys.path.append('../')
from LSTM import data_LSTM, LSTM_1
import numpy as np

data = pd.read_pickle('data/NEW_wPLI_all_10_1_left_alpha.pickle')
data=data.query("Phase=='Base'")

part = ['13', '18', '05', '11', '19', '02', '20', '22', '12', '10', '09']

Part_chro=['13','22','10', '18','05','12','11']
Part_reco=['19','20','02','09']
stepsize_r=12
stepsize_c=20
windowsize=20

dataset, ID= data_LSTM.prepare_data_LSTM(data=data,
                                         Part_chro=Part_chro,
                                         Part_reco=Part_reco,
                                         stepsize_c=stepsize_c,
                                         stepsize_r=stepsize_r,
                                         windowsize=windowsize)

len(np.where(np.array(dataset.labels)==1)[0])
len(np.where(np.array(dataset.labels)==0)[0])

splits=[int(len(dataset)*0.8), int(len(dataset))-int(len(dataset)*0.8)]
train_set,dev_set=torch.utils.data.random_split(dataset, splits)

lrs=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5]
bs=[2,3,4,5,6,7,8,9]
hd=[2,3,4,5,7,8,9,10]

acc_lrs=[]
loss_lrs=[]

#for lr in lrs:
#for b in bs:
for h in hd:
    batch_size = 6
    hidden_dim = 4 #h
    learning_rate = 0.01 #lr
    num_layers = 1
    nr_epochs = 10

    correct_values, loss_values, dev_acc, model= LSTM_1.train_LSTM(train_set=train_set,
                                                                   dev_set=dev_set,
                                                                   batch_size=batch_size,
                                                                   hidden_dim=hidden_dim,
                                                                   learning_rate=learning_rate,
                                                                   num_layers=num_layers,
                                                                   nr_epochs=nr_epochs)
    acc_lrs.append(dev_acc)
    loss_lrs.append(loss_values)


plt.plot(np.transpose(pd.DataFrame(loss_lrs)))
plt.legend(hd)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()
#plt.savefig('learning_rate.png')
plt.savefig('batch_size.png')

plt.plot(np.transpose(pd.DataFrame(acc_lrs)))
plt.legend(hd)
plt.ylabel('dev_accuracy (30% dev set)')
plt.xlabel('epochs')
plt.show()