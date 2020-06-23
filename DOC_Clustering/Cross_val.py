import pandas as pd
import torch
torch.manual_seed(1)
import sys
sys.path.append('../')
import data_LSTM
import LSTM_1
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_pickle('data/NEW_wPLI_all_10_1_left_alpha.pickle')
data=data.query("Phase=='Anes'")

# ALL Hyperparameters
batch_size = 5
hidden_dim = 5
learning_rate = 0.05
num_layers = 1
nr_epochs = 10
Part_chro=['13','22','10', '18','05','12','11']
Part_reco=['19','20','02','09']


part = ['13', '18', '05', '11', '19', '02', '20', '22', '12', '10', '09']
cv_acc = {}
cv_pred = {}
cv_corr = {}

for p in range(len(part)):
    part_tst = [part[p]]
    part_tng=part.copy()
    part_tng.remove(part_tst[0])

    data_train=data[data['ID'].isin(part_tng)]
    data_test=data[data['ID'].isin(part_tst)]

    # if participant from recovered group
    if Part_reco.count(part_tst[0]) ==1:
        stepsize_r = 10 # 80/90 ratio
    else:
        stepsize_r = 13 #80/80 ratio

    stepsize_c = 20
    windowsize = 20

    dataset_train, ID_train=data_LSTM.prepare_data_LSTM(data=data_train,
                                Part_chro=Part_chro,
                                Part_reco=Part_reco,
                                stepsize_c=stepsize_c,
                                stepsize_r=stepsize_r,
                                windowsize=windowsize)

    #len(np.where(np.array(dataset_train.labels)==1)[0])
    #len(np.where(np.array(dataset_train.labels)==0)[0])

    dataset_test, ID_test=data_LSTM.prepare_data_LSTM(data=data_test,
                                Part_chro=Part_chro,
                                Part_reco=Part_reco,
                                stepsize_c=stepsize_c,
                                stepsize_r=stepsize_r,
                                windowsize=windowsize)


    correct_values, loss_values, dev_acc, model=LSTM_1.train_LSTM(train_set=dataset_train,
                                                                  dev_set=False,
                                                                  batch_size=batch_size,
                                                                  hidden_dim=hidden_dim,
                                                                  learning_rate=learning_rate,
                                                                  num_layers=num_layers,
                                                                  nr_epochs=nr_epochs)


    predicted, right, accuracy = LSTM_1.test_LSTM(test_set=dataset_test,
                                           model=model)

    cv_acc[part_tst[0]]=accuracy
    cv_corr[part_tst[0]]=right
    cv_pred[part_tst[0]]=predicted



# Figure for recovered Patients
fig, axes = plt.subplots(nrows=len(Part_reco), ncols=1)
plt.setp(axes, xticks=[], xticklabels=[],yticks=[1])
fig.suptitle("Recovered Patients")
for p in range(len(Part_reco)):
    a=cv_pred[Part_reco[p]]==cv_corr[Part_reco[p]]
    # Use the pyplot interface to change just one subplot...
    a = [int(elem) for elem in a]
    a = np.array(a).reshape(1, -1)
    plt.sca(axes[p])
    plt.imshow(a,cmap='prism',vmin=0,vmax=1)
    plt.yticks(range(1), [Part_reco[p]])

fig.tight_layout()
plt.show()

# Figure for chronic Patients
fig, axes = plt.subplots(nrows=len(Part_chro), ncols=1)
plt.setp(axes, xticks=[], xticklabels=[],yticks=[1])
fig.suptitle("Chronic Patients")
for p in range(len(Part_chro)):
    a=cv_pred[Part_chro[p]]==cv_corr[Part_chro[p]]
    # Use the pyplot interface to change just one subplot...
    plt.sca(axes[p])
    a = np.array([int(elem) for elem in a])
    a = a.reshape(1, -1)
    plt.imshow(a,cmap='prism',vmin=0,vmax=1)
    plt.yticks(range(1), [Part_chro[p]])

fig.tight_layout()
plt.show()

plt.plot(*zip(*sorted(cv_acc.items())))
plt.show()


cv_pred['11']
cv_acc['05']
cv_corr['11']