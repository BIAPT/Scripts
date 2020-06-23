import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
torch.manual_seed(1)
import sys
sys.path.append('../')
import data_LSTM
import LSTM_1

data = pd.read_pickle('data/NEW_wPLI_all_10_1_left_alpha.pickle')
data=data.query("Phase=='Base'")

part = ['13', '18', '05', '11', '19', '02', '20', '22', '12', '10', '09']

Part_chro=['13','22','10', '18','05','12','11']
Part_reco=['19','20','02','09']
stepsize_r=12
stepsize_c=20
windowsize=20

dataset, ID=data_LSTM.prepare_data_LSTM(data=data,
                            Part_chro=Part_chro,
                            Part_reco=Part_reco,
                            stepsize_c=stepsize_c,
                            stepsize_r=stepsize_r,
                            windowsize=windowsize)

splits=[int(len(dataset)*0.8), int(len(dataset))-int(len(dataset)*0.8)]
train_set,dev_set=torch.utils.data.random_split(dataset, splits)

lrs=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5]
acc_lrs=[]
loss_lrs=[]

for lr in lrs:
    batch_size = 5
    hidden_dim = 10
    learning_rate = lr
    num_layers = 1
    nr_epochs = 3

    correct_values, loss_values, dev_acc, model=LSTM_1.train_LSTM(train_set=train_set,
                                                                  dev_set=dev_set,
                                                                  batch_size=batch_size,
                                                                  hidden_dim=hidden_dim,
                                                                  learning_rate=learning_rate,
                                                                  num_layers=num_layers,
                                                                  nr_epochs=nr_epochs)
    acc_lrs.append(dev_acc)
    loss_lrs.append(loss_values)

for i in range(len(lrs)):
    plt.plot(loss_lrs[i])
plt.legend(lrs)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

for i in range(len(lrs)):
    plt.plot(acc_lrs[i])
plt.legend(lrs)
plt.ylabel('dev_accuracy (30% dev set)')
plt.xlabel('epochs')
plt.show()