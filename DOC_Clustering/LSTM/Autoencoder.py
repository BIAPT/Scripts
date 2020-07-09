import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import torch
torch.manual_seed(1)
import sys
sys.path.append('../')
from LSTM import data_LSTM, LSTM_Auto
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#data = pd.read_pickle('../data/WholeBrain_wPLI_10_1_alpha.pickle')
data = pd.read_pickle('data/WholeBrain_wPLI_10_1_alpha.pickle')
#data = pd.read_pickle('data/WholeBrain_dPLI_10_1_alpha.pickle')

areas=data.columns[4:]
Phase='combined'

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
                                         windowsize=windowsize,
                                         Phase=Phase)

len(np.where(np.array(dataset.labels)==1)[0])
len(np.where(np.array(dataset.labels)==0)[0])

#splits=[int(len(dataset)*0.8), int(len(dataset))-int(len(dataset)*0.8)]
#train_set,dev_set=torch.utils.data.random_split(dataset, splits)

batch_size = 10
hidden_dim = 6
learning_rate = 0.1  # lr
num_layers = 1
nr_epochs = 30

loss_values, dev_loss, model = LSTM_Auto.train_LSTM(train_set=dataset,
                                                                dev_set=dataset,
                                                                batch_size=batch_size,
                                                                hidden_dim=hidden_dim,
                                                                learning_rate=learning_rate,
                                                                num_layers=num_layers,
                                                                nr_epochs=nr_epochs,
                                                                input_dim=55)

predicted, right, latent, label, loss, h_states = LSTM_Auto.test_LSTM(test_set=dataset, model=model)

input=np.zeros([1,40,hidden_dim])
input[:,:,0]=1 # visualize 1st latent space
input=torch.tensor(input, dtype= torch.float )

pred,_ = model.lstm2(input)
pred2=pred.detach().numpy()

dim_0=np.mean(pred2[0],0)
plt.plot(np.mean(pred2[0],0))
plt.xticks(range(55), areas, rotation='vertical')

data=pd.read_pickle('data/WholeBrain_wPLI_10_1_alpha.pickle')
data_A = data.query("Phase=='Anes'")
data_B = data.query("Phase=='Base'")
areas=data.columns[4:]
Part=['13','22','10', '18','05','12','11','19','20','02','09']

max_len=190
time_data_Base=np.zeros([len(Part),max_len,1])
time_data_Anes=np.zeros([len(Part),max_len,1])

dim_0=dim_0.reshape(55,1)

for p in range(len(Part)):
    tmp = data_B.query("ID=='{}'".format(Part[p]))
    time_data_Base[p]=tmp.iloc[:max_len,4:].dot(dim_0)
    tmp = data_A.query("ID=='{}'".format(Part[p]))
    time_data_Anes[p]=tmp.iloc[:max_len,4:].dot(dim_0)

plt.figure()
plt.plot(np.transpose(np.mean(np.mean(time_data_Base[7:],2),0)),label='recovered')
plt.plot(np.transpose(np.mean(np.mean(time_data_Base[:7],2),0)),label='non-recovered')
plt.title('Baseline')
plt.legend()
plt.ylim(0,1)
plt.figure()
plt.plot(np.transpose(np.mean(np.mean(time_data_Anes[7:],2),0)),label='recovered')
plt.plot(np.transpose(np.mean(np.mean(time_data_Anes[:7],2),0)),label='non-recovered')
plt.title('Anesthesia')
plt.ylim(0,1)
plt.legend()




plt.plot(np.mean(np.mean(right[np.where(label == 0)[0]],0),1))
plt.plot(np.mean(np.mean(predicted[np.where(label == 0)[0]],0),1))
plt.legend(['right','predicted'])
plt.title('non-recovered')
plt.show()

plt.figure()
plt.plot(np.mean(np.mean(right[np.where(label == 1)[0]],0),1))
plt.plot(np.mean(np.mean(predicted[np.where(label == 1)[0]],0),1))
plt.legend(['right','predicted'])
plt.title('recovered')
plt.show()


plt.plot(np.mean(np.array(h_states)[0,np.where(label==1)[0],:],0))
plt.plot(np.mean(np.array(h_states)[0,np.where(label==0)[0],:],0))

plt.figure()
for i in range(hidden_dim):
    plt.plot(np.mean(latent[np.where(label == 1)[0]], 0)[:, i],label=str(i))
plt.legend()
plt.title('recovered patients')
plt.ylim(-0.8,0.8)

plt.figure()
for i in range(hidden_dim):
    plt.plot(np.mean(latent[np.where(label == 0)[0]], 0)[:, i],label=str(i))
plt.legend()
plt.title('non recovered patients')
plt.ylim(-0.8,0.8)



fig = plt.figure()
ax = Axes3D(fig)

plt.figure()
for i in (np.where(label==1)[0]):
    #plt.scatter(np.array(latent[i][:,0]), np.array(latent[i][:,1]), np.array(latent[i][:,2]), edgecolors='blue')
    #plt.scatter(np.array(latent[i][:,0]), np.array(latent[i][:,1]),range(40), edgecolors='blue')
    plt.scatter( range(40),np.array(latent[i][:,0]),range(40), edgecolors='blue')
    #plt.plot(np.array(latent[i][:,0]), color='blue')

plt.figure()
for i in (np.where(label==0)[0]):
    #plt.scatter(np.array(latent[i][:,0]), np.array(latent[i][:,1]), np.array(latent[i][:,2]), edgecolors='red')
    #plt.scatter(np.array(latent[i][:,0]), np.array(latent[i][:,1]),range(40), edgecolors='red')
    plt.scatter( range(40),np.array(latent[i][:,0]),range(40), edgecolors='red')
    #plt.plot(np.array(latent[i][:,0]), color='red')
plt.legend(['recovered', 'non-recovered'])

