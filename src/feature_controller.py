import os
import sys
import shutil
import GPUtil
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = dir_path+'/../model/'
DATA_PATH = dir_path+'/../data/'

import src.util as util
_SEED = 42
util.set_seed(_SEED)

class ControllerNet(nn.Module):
    def __init__(self, _num_of_features, _lstmcell_hidden_size):
        super(ControllerNet, self).__init__()

        # Num of features correspond to the number of lstm cells 
        # each nn.LSTM consists of a single lstm cell, we use nn.LSTM instead of nn.LSTMCell 
        # because it is CUDA optimized

        self.num_of_features = _num_of_features
        self.lstmcell_hidden_size = _lstmcell_hidden_size
        self.lstm_feat = []
        self.dense_feat = []

        for i in range(self.num_of_features):
            self.lstm_feat.append(nn.LSTM(input_size=self.lstmcell_hidden_size, hidden_size=self.lstmcell_hidden_size, num_layers=1, batch_first=True))
            self.dense_feat.append(nn.Linear(self.lstmcell_hidden_size, 1))

        self.lstm_feat = nn.ModuleList(self.lstm_feat)
        self.dense_feat = nn.ModuleList(self.dense_feat)

    def forward(self, x, state):
        output = []
        (h0, c0) = state
        for i in range(self.num_of_features):
            x, (h0, c0) = self.lstm_feat[i](x, (h0, c0))
            x = x.permute(1,0,2)
            output.append(F.sigmoid(self.dense_feat[i](h0[-1])))

        return output, (h0, c0)

class Controller:
    def __init__(self,
                 lstm_hidden_size=6,
                 output_dimension=1,
                 num_features=6,
                 baseline_decay=0.999,
                 epochs = 20,
                 batch_size = 1,
                 custom_filename = ''):

        self.lstm_hidden_size = lstm_hidden_size
        self.output_dimension = output_dimension
        self.num_features = num_features
        self.baseline_reward = None
        self.reward = 0
        self.baseline_decay = baseline_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_mean = 40
        self.action_memory = []
        self.reward_memory = []
      
        # build controller here # fix the optimizer
        if torch.cuda.is_available():
            self.controller_net = ControllerNet(self.num_features, self.lstm_hidden_size).cuda()
        else:
            self.controller_net = ControllerNet(self.num_features, self.lstm_hidden_size)

        # zero the parameter gradients
        self.optimizer = optim.Adam(self.controller_net.parameters(), lr=0.001)
        self.optimizer.zero_grad()

        # Load controller here if found checkpoint
        self.root_dir = MODEL_PATH
        self.model_filename = "controller_agent_"+custom_filename+".pt"
        self.model_location = self.root_dir+self.model_filename
        if self.model_location is not None and os.path.exists(self.model_location):
            self.controller_net, self.optimizer = self.load_model(self.controller_net, self.optimizer)

    def get_action(self, state):
        # with torch.no_grad():

        (h0, c0) = state 
        input = state[0]

        probabilities = None
        if torch.cuda.is_available():
            probabilities, (hn, cn) = self.controller_net(input.float().cuda(), (h0.float().cuda(), c0.float().cuda()))
        else:
            probabilities, (hn, cn) = self.controller_net(input.float(), (h0.float(), c0.float()))

        action_list = []
        sampled_action = []
        
        for _p in probabilities:
            # make a decision based on the probability if you want to choose a particular action or not
            individual_action = np.random.binomial(n=1, p=_p.cpu().detach().numpy())
            action_list.append(individual_action[0][0])
            sampled_action.append(torch.from_numpy(individual_action))
            
        action_list = np.asarray(action_list, dtype=np.int)
        print(probabilities)
        return action_list, sampled_action, probabilities, (hn, cn)

    def controller_loss(self, output, target):
        # create empty loss list
        loss = [None] * len(target)
        _log = [None] * len(target)
        log_lik = [None] * len(target)
        
        for i in range(len(target)):
            _log[i] = torch.log(torch.clamp(output[i], 1e-8, 1-1e-8)).cuda().float()
            log_lik[i] = target[i].cuda().float()*_log[i]
            if(len(self.reward_memory) >= 1):
                loss[i] = torch.sum(-log_lik[i]*(self.get_reward() - np.mean(self.reward_memory)))
            else:
                loss[i] = torch.sum(-log_lik[i]*(self.get_reward()))

        return loss

    def train_controller(self, state, target):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        for epoch in range(self.epochs):
            # zero the parameter gradients
            self.optimizer.zero_grad()
            
            (h0, c0) = state
            input = h0
            
            # with torch.autograd.set_detect_anomaly(True):
            # forward + backward + optimize
            if torch.cuda.is_available():
                output, (hn, cn) = self.controller_net(input.float().cuda(), (h0.float().cuda(), c0.float().cuda()))
                loss = self.controller_loss(output, target)
            else:
                output, (hn, cn) = self.controller_net(input.float(), (h0.float(), c0.float()))
                loss = self.controller_loss(output, target)
            
            # loss.backward(retain_graph=True)
            torch.autograd.backward(loss, retain_graph=True)

            self.optimizer.step()
            # running_loss += loss.item()
            
        # calculate auc_here/acc for validation set here
        print("Moving average : ", str(self.get_reward() if len(self.reward_memory)<1 else np.mean(self.reward_memory)))
    
    def store_transition(self, action, reward):
        self.action_memory.append(action)
        self.reward_memory.append(reward)        

    def set_buffer(self, memory):
        self.action_memory = memory["action_memory"]
        self.reward_memory = memory["reward_memory"]

    def get_buffer(self):
        memory = {"action_memory" : self.action_memory,
                  "reward_memory" : self.reward_memory}
        return memory
    
    def get_reward(self):
        return self.reward

    def set_reward(self, reward):
        self.reward = reward
    
    def save_model(self):
        state = {
            'state_dict': self.controller_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, self.root_dir+self.model_filename)

    def load_model(self, model, optimizer):
        checkpoint = torch.load(self.root_dir+self.model_filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer