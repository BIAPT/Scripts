from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(1)
import sys
import numpy as np
sys.path.append('../')


class MyLSTM(nn.Module):
    def __init__(self, hidden_dim, feature_size, target_size, num_layers):
        super(MyLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(feature_size, hidden_dim, batch_first=True, num_layers=num_layers)
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        tag_space = self.hidden2tag(lstm_out[:, -1, :])
        #tag_pred = torch.sigmoid(tag_space)
        tag_pred = tag_space

        return tag_pred


def train_LSTM (train_set, dev_set, batch_size, hidden_dim, learning_rate, num_layers, nr_epochs):

    # Prepare Dataloader
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    if dev_set != False:
        dataloader_dev = DataLoader(dev_set, batch_size=len(dev_set), shuffle=False)

    # intern parameters
    feature_size = 15
    target_size = 2

    # initialize model
    model = MyLSTM(hidden_dim, feature_size, target_size, num_layers)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # initialize empty output values
    loss_values = []
    correct_values = []
    dev_acc = []

    # train all epochs
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
        accuracy = (correct / len(train_set)) * 100
        correct_values.append(accuracy)
        print('epoch {} accuracy: {}'.format(epoch, accuracy))

        if dev_set != False:
            with torch.no_grad():
                for i, data in enumerate(dataloader_dev, 0):
                    label = data[1]
                    input_ = data[0]
                    tag_scores = model(input_)
                    predicted = torch.max(tag_scores, 1)[1]
                    correct = (predicted == label.view(1, len(label))).sum().item()
                    accuracy = correct / len(label)*100
                    dev_acc.append(accuracy)

    return correct_values, loss_values, dev_acc, model


def test_LSTM(test_set, model):
    dataloader_test = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    with torch.no_grad():
        for i, data in enumerate(dataloader_test, 0):
            label = data[1]
            input_ = data[0]
            tag_scores = model(input_)
            pred = (torch.max(tag_scores, 1))[1]
            predicted = pred.view(len(pred))
            right = label.reshape(len(label))
            correct = (pred == right).sum().item()
            accuracy=(correct/len(predicted))*100
            print('right : ')
            print(right)
            print('predicted :')
            print(predicted)
            print('accuracy : {}'.format(accuracy))

            return np.array(predicted), np.array(right), accuracy
