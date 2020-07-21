import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(1)
import sys
import os
sys.path.append('../')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

data = pd.read_pickle('data/NEW_wPLI_all_10_1_left_alpha.pickle')
data=data.query("Phase=='Base'")

Part_chro=['13','22','10', '18','05','12','11']
Part_reco=['19','20','02','09']

plt.plot(Part_chro)
plt.show()
#part_tng=['13','18','05','11','19','02','20','22']
#part_tst=['09','12','10']

data_sample = []
stepsize_r=12
stepsize_c=20
windowsize=20
labels=[]
ID=[]
s=0
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
ID=pd.DataFrame(ID)

labels = pd.DataFrame(labels).astype('category')
encode_map = {'r': 1, 'c': 0}
labels.replace(encode_map, inplace=True)
labels=labels.values.tolist()

areas = ['FC', 'FP', 'FO', 'FT', 'TO', 'TC', 'TP', 'PO', 'PC', 'CO', 'FF', 'CC', 'PP', 'TT', 'OO']


#data_sample_tng=data_sample[data_sample['ID'].isin(part_tng)]
#data_sample_tst=data_sample[data_sample['ID'].isin(part_tst)]
#labels_tng=pd.DataFrame(labels)[ID[0].isin(part_tng)].values.tolist()
#labels_tst=pd.DataFrame(labels)[ID[0].isin(part_tst)].values.tolist()


data_by_sample = list(data_sample.query("Phase=='Base'").groupby('sample'))
data_by_sample = [dd[1][areas].values for dd in data_by_sample]

#data_by_sample_tng = list(data_sample_tng.query("Phase=='Base'").groupby('sample'))
#data_train = [dd[1][areas].values for dd in data_by_sample_tng]

#data_by_sample_tst = list(data_sample_tst.query("Phase=='Base'").groupby('sample'))
#data_test = [dd[1][areas].values for dd in data_by_sample_tst]

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = [torch.tensor(ll, dtype=torch.long) for ll in labels]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset(data_by_sample, labels)
DATASIZE=int(len(dataset))

splits=[int(len(dataset)*0.8), int(len(dataset))-int(len(dataset)*0.8)]
train_set,test_set=torch.utils.data.random_split(dataset, splits)

'''
##########################################
###         METHOD MODEL              ###
##########################################
'''

def perform_LSTM (train_set, test_set, BATCH_size, hidden_dim, LEARNING_rate, nr_epochs):

    dataloader_train = DataLoader(train_set, batch_size=BATCH_size, shuffle=True)
    dataloader_test = DataLoader(test_set, batch_size=len(test_set), shuffle=True)


    #tng_dataset = MyDataset(data_train, labels_tng)
    #tng_dataloader = DataLoader(tng_dataset, batch_size=BATCH_size, shuffle=True)

    #tst_dataset = MyDataset(data_test, labels_tst)
    #tst_dataloader = DataLoader(tst_dataset,batch_size=len(tst_dataset))


    feature_size=15
    target_size=2

    class MyLSTM(nn.Module):
        def __init__(self, hidden_dim, feature_size, target_size):
            super(MyLSTM,self).__init__()
            self.hidden_dim = hidden_dim
            #self.lstm = nn.LSTM(hidden_size=hidden_dim, num_layers=1, input_size=15, batch_first=True)
            self.lstm = nn.LSTM(feature_size,hidden_dim, batch_first=True)
            self.hidden2tag = nn.Linear(hidden_dim, target_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            #print("lstmout", lstm_out[:, -1, :].shape)
            tag_space = self.hidden2tag(lstm_out[:,-1, :])
            #print("tagspace", tag_space.shape)
            #tag_pred = torch.sigmoid(tag_space)
            tag_pred = tag_space
            #print("tagspred", tag_pred)

            return tag_pred

    model = MyLSTM(hidden_dim, feature_size, target_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_rate)

    #def training(nr_epochs, data_train,data_test):
    loss_values=[]
    correct_values=[]
    cv_acc=[]
    for epoch in range(nr_epochs):
        running_loss = 0
        correct = 0
        for i, data in enumerate(dataloader_train, 0):
            label_ = data[1]
            input_ = data[0]

            model.zero_grad()
            tag_score = model(input_)
            loss = loss_function(tag_score, torch.max(label_, 1)[0])
            loss.backward()
            optimizer.step()
            # Append Loss
            running_loss += loss.item()
            _, predicted = torch.max(model(input_), 1)
            correct += (predicted == (torch.max(label_, 1)[0])).sum().item()
        loss_values.append(running_loss / len(dataloader_train)*100)
        accuracy = (correct / DATASIZE) * 100
        correct_values.append(accuracy)
        print('epoch {} accuracy: {}'.format(epoch,accuracy) )
        with torch.no_grad():
            for i, data in enumerate(dataloader_test, 0):
                label = data[1]
                input_ = data[0]
                tag_scores = model(input_)
                predicted = torch.max(tag_scores, 1)[1]
                correct = (predicted == label.view(1, len(label))).sum().item()
                accuracy = correct / len(label)*100
                cv_acc.append(accuracy)

    return correct_values,loss_values,cv_acc


'''
##########################################
###         PERFORM MODEL              ###
##########################################
'''
BATCH_size = 2
hidden_dim = 10
LEARNING_rate = 0.01
EPOCH_nr = 100
lrs=[0.0001,0.001,0.005,0.01,0.05,0.1]
hd=[3,5,7,9,11]
loss_lrs=[]
acc_lrs=[]

for hp in lrs:
    LEARNING_rate=hp
    hidden_dim=5
    correct_values,loss_values,cv_acc = perform_LSTM(train_set=train_set,test_set=test_set,
                                                     nr_epochs= EPOCH_nr,
                                                     BATCH_size=BATCH_size,
                                                     LEARNING_rate=LEARNING_rate,
                                                     hidden_dim=hidden_dim)
    acc_lrs.append(cv_acc)
    loss_lrs.append(loss_values)



#plt.plot(correct_values)
plt.plot(loss_lrs[0])
plt.plot(loss_lrs[1])
plt.plot(loss_lrs[2])
plt.plot(loss_lrs[3])
plt.plot(loss_lrs[4])
plt.plot(loss_lrs[5])
plt.legend(lrs)
plt.show()

#plt.plot(cv_acc)
#plt.legend(['accuracy','loss','cv'])
#plt.show()

def testing(data_test):
    with torch.no_grad():
        for i, data in enumerate(data_test, 0):
            label = data[1]
            input_ = data[0]
            tag_scores = model(input_)
            predicted = torch.max(tag_scores, 1)[1]
            correct = (predicted == label.view(1,len(label))).sum().item()
            accuracy=correct/len(label)
            print('right : ')
            print(label.view(1,len(label)))
            print('predicted :')
            print(predicted.view(1,len(predicted)))
            print('accuracy : {}'.format(accuracy))


testing(dataloader_test)
