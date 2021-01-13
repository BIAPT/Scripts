from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(1)
import sys
import numpy as np
sys.path.append('../')


class MyAutoLSTM(nn.Module):
    def __init__(self, hidden_dim, feature_size, target_size, num_layers):
        super(MyAutoLSTM, self).__init__()
        self.lstm1 = nn.LSTM(feature_size, hidden_dim, batch_first=True, num_layers=num_layers)
        self.lstm2 = nn.LSTM(hidden_dim, feature_size, batch_first=True, num_layers=num_layers)
        #self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, x):
        latent, (h_state, _) = self.lstm1(x)
        lstm_out, _ = self.lstm2(latent)

        return lstm_out, latent, h_state


def train_LSTM (train_set, dev_set, batch_size, hidden_dim, learning_rate, num_layers, nr_epochs, input_dim):

    # Prepare Dataloader
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    if dev_set != False:
        dataloader_dev = DataLoader(dev_set, batch_size=len(dev_set), shuffle=False)

    # intern parameters
    feature_size = input_dim
    target_size = input_dim

    # initialize model
    model = MyAutoLSTM(hidden_dim, feature_size, target_size, num_layers)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # initialize empty output values
    loss_values = []
    correct_values = []
    dev_acc = []

    # train all epochs
    for epoch in range(nr_epochs):
        running_loss = 0
        for i, data in enumerate(dataloader_train, 0):
            label_ = data[1]
            input_ = data[0]

            model.zero_grad()
            predicted, _, _ = model(input_)
            loss = loss_function(predicted, input_)
            loss.backward()
            optimizer.step()
            # Append Loss
            running_loss += loss.item()

        loss_values.append(running_loss / len(dataloader_train)*100)
        print('epoch {} loss: {}'.format(epoch, running_loss / len(dataloader_train)*100))

        if dev_set != False:
            with torch.no_grad():
                dev_running_loss = 0
                for i, data in enumerate(dataloader_dev, 0):
                    dev_input_ = data[0]
                    dev_predicted, _, _ = model(dev_input_)
                    dev_loss = loss_function(dev_predicted, dev_input_)
                    dev_running_loss += dev_loss.item()
            print('epoch {} dev_loss: {}'.format(epoch, dev_running_loss / len(dataloader_dev) * 100))

    return loss_values, dev_loss, model


def test_LSTM(test_set, model):
    dataloader_test = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    loss_function = nn.MSELoss()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test, 0):
            label = data[1]
            input_ = data[0]
            pred, latent, h_state = model(input_)
            loss = loss_function(pred, input_)
            print('loss :')
            print(loss)

            return np.array(pred), np.array(input_), np.array(latent), label, loss, np.array(h_state)
