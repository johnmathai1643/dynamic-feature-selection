import os
import sys
import GPUtil
import gc
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

import src.util as util
_SEED = 42
util.set_seed(_SEED)

class ChildNetwork:
    def __init__(self, X_train, Y_train, X_val, Y_val,
                 _test_split=0.2,
                 batch_size=32,
                 patience = 10,
                 epoch=50,
                 _random_state=_SEED,
                 is_timeseries=True,
                 default_gpu="cuda:0"):

        self.X_train = X_train
        self.Y_train = Y_train

        self.X_val = X_val
        self.Y_val = Y_val

        self.batch_size = batch_size
        self.patience = patience
        self.epoch = epoch
        self.is_timeseries = is_timeseries
        self.device = torch.device(default_gpu)

    class DataPipeline(data.Dataset):
        x = None
        y = None

        def __init__(self, x, y, action):
            timestep = x.shape[1]
            
            # TX = np.repeat(np.array(action)[np.newaxis,:], timestep, 0)
            # for i in range(x.shape[0]):
            #     x[i] = np.multiply(x[i], TX)

            action = np.array(action)
            _x = np.empty((x.shape[0], timestep, np.count_nonzero(action)))
            col_index = np.where((action).astype(bool)==False)

            for i in range(x.shape[0]):
                _x[i] = np.delete(x[i], col_index, 1)
         
            y = np.reshape(y, (len(y), 1))
            
            self.x = _x
            self.y = y
                
        def __getitem__(self, index):
            _x = torch.from_numpy(self.x[index])
            _y = torch.from_numpy(self.y[index])
            return _x, _y

        # Override to give PyTorch size of dataset
        def __len__(self):
            return len(self.x)

    class ChildNet(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.conv1 = nn.Conv1d(3, 64, 1)
            self.conv1_bn = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 64, 1)
            self.conv2_bn = nn.BatchNorm1d(64)
            self.lstm_1 = nn.LSTM(input_size=features, hidden_size=32, num_layers=1, batch_first=True)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            self.sigmoid = nn.Sigmoid()
            
        def init_hidden(self, batch_size):
            # This is what we'll initialise our hidden state as, # hidden state is output of RNN
            # num_layers, batch_size, hidden_state
            return Variable(torch.zeros(1, batch_size, 32))
        
        def forward(self, x):
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = F.relu(self.conv2_bn(self.conv2(x)))
            lstm_out,_ = self.lstm_1(x)
            lstm_out = lstm_out.permute(1,0,2)
            x = F.relu(self.fc2(lstm_out[-1,:,:].view(-1, lstm_out.shape[2])))
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x 

    def build_network(self, features):
        return self.ChildNet(features).to(self.device)

    def step(self, action):
        
        # action = np.ones(self.X_train.shape[2])

        if(np.count_nonzero(action)<=0):
            reward = -1.0
            return action, reward

        train_loader = DataLoader(self.DataPipeline(self.X_train, self.Y_train, action), batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=True)
        val_loader = DataLoader(self.DataPipeline(self.X_val, self.Y_val, action), batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        net = self.build_network(np.count_nonzero(action))
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
        # optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

        # patience counter for early stopping
        epoch_patience = 0
        min_val_loss = float("inf")
        # holds validation results over all epochs
        net_history = []
        # loop over the dataset multiple times
        for epoch in range(self.epoch):
            running_loss = 0.0

            for i, data in enumerate(train_loader):
                # zero the parameter gradients
                optimizer.zero_grad()
                # Clear hidden states         
                net.hidden = net.init_hidden(self.batch_size)
        
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
               
                # forward + backward + optimize
                outputs = net(inputs.float().to(self.device))
                loss = criterion(outputs.float().to(self.device), labels.float().to(self.device))
                
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            
            # calculate auc_here/acc for validation set here
            cur_val_loss, val_auc = self.validation_score(net, val_loader, metric="aucroc")
            net_history.append(val_auc)
            print('Child Epoch [%d] loss: %.3f val_loss: %.3f val_auc: %.3f' % (epoch + 1, running_loss, cur_val_loss, val_auc))

            if cur_val_loss <= min_val_loss:
                min_val_loss = cur_val_loss
                epoch_patience = 0
            elif self.patience == epoch_patience:
                break
            else:
                epoch_patience = epoch_patience + 1

        # net history contains the accuracy/auc_roc of validation models
        net_history.sort()
        reward = 0
        for i in range(1, min(len(net_history), 5)):
            reward+= pow(float(net_history[-i]), 4)
        
        reward = reward - (np.count_nonzero(action)/np.shape(action)[0])

        self.garbage_collect(net)
        return action, reward
            
    def validation_score(self, net, val_loader, metric = "accuracy"):
        # supported metrics accuracy, aucroc
        y_true = []
        y_pred = []

        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        net.eval()
        running_loss = 0.0
        criterion = nn.BCELoss()

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                
                # Clear hidden states         
                net.hidden = net.init_hidden(1)

                if torch.cuda.is_available():
                    outputs = net(inputs.float().to(self.device))
                    running_loss += criterion(outputs.float().to(self.device), labels.float().to(self.device))
                
                y_true.append(labels.numpy()[0][0])
                y_pred.append(torch.round(outputs.cpu()).numpy()[0][0])

        net.train()

        if metric == "aucroc":
            return running_loss.detach().cpu().numpy(), roc_auc_score(y_true, y_pred)

        # default return accuracy
        return running_loss.numpy(), accuracy_score(y_true, y_pred)

    def predict(self):
        test_loader = DataLoader(DataPipeline('test'), batch_size=1, shuffle=False, num_workers=0, drop_last=False)
       
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                if torch.cuda.is_available():
                    outputs = net(inputs.float().to(self.device))
                else:
                    outputs = net(inputs.float())
                y_true.append(labels.numpy()[0][0])
                y_pred.append(torch.round(outputs.cpu()).numpy()[0][0])

        return accuracy_score(y_true, y_pred)

    def garbage_collect(self, net):
        del net
        gc.collect()
        torch.cuda.empty_cache()
