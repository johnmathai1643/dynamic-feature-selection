import os
import numpy as np
import random
import gc
import pickle

import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 

dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = dir_path+'/model/'
DATA_PATH = dir_path+'/data/'

# import custom py files and class here
import src.util as util
# from src.util import roc_callback
from src.feature_controller import Controller
from src.child_network import ChildNetwork

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

_SEED = 42
util.set_seed(_SEED)

class DynamicSelection():
    def __init__(self, x_train, y_train, x_test, y_test, epoch, random_state=_SEED, load_previous = True, custom_filename=''):
        self.filename = MODEL_PATH+"training_state_"+custom_run_name+".pickle"
        self.custom_filename = custom_filename
        self.start_epoch = 0
        self.epoch = epoch

        # initialize feature controller
        self.lstm_hidden_size = 4
        # contains the input, short term, long term state of the controller
        self.cur_state = (torch.rand(1, 1, self.lstm_hidden_size),  torch.rand(1, 1, self.lstm_hidden_size))
        self.agent = Controller(lstm_hidden_size = self.lstm_hidden_size, num_features = x_train.shape[2], custom_filename=self.custom_filename)

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(x_train, y_train, test_size=0.15, stratify = y_train, random_state=random_state)
        self.X_test = x_test
        self.Y_test = y_test

        # load previous training state epochs, state_vector, prob_vector, action, storage buffers
        if self.filename is not None and os.path.exists(self.filename) and load_previous:
            state_var_dict = self.load_state(self.filename)
            self.start_epoch = state_var_dict["start_epoch"]
            self.cur_state = state_var_dict["lstm_state"]
            self.agent.set_buffer(state_var_dict["memory"])

    def save_state(self, filename, agent, i, lstm_state):
        state_var_dict = {"memory" : agent.get_buffer(), "start_epoch" : i, "lstm_state": lstm_state}
        with open(filename, 'wb') as f:
            pickle.dump(state_var_dict, f)

    def load_state(self, filename):
        with open(filename, 'rb') as f:
            state_var_dict = pickle.load(f)
        return state_var_dict

    def run(self):
        for i in range(self.start_epoch, self.epoch):
            print("Epoch : " + str(i))
            action, sampled_action, pred_prob, next_state = self.agent.get_action(self.cur_state)

            child_network = ChildNetwork(self.X_train, self.Y_train, self.X_val, self.Y_val, feature_num)
            action, reward = child_network.step(action)

            self.agent.set_reward(reward)
            print("Current action : " + str(action))
            print("Features : " + str(np.count_nonzero(action)))
            print("Reward : "  + str(self.agent.get_reward()))
            self.agent.train_controller(self.cur_state, sampled_action)
            self.agent.store_transition(action, reward)
            
            # assign next cur_state # batch_size is always one
            self.cur_state = next_state
            del child_network

            if (i%3==0):
                self.save_state(self.filename, self.agent, i, next_state)
                self.agent.save_model()

            print("------------------------------------------------------------------------------")
    
        score_history = self.agent.get_buffer()["reward_memory"]
        util.plotLearning(score_history, filename="dynamic_feature_selection_"+self.custom_filename+".png", window=100)

if __name__ == '__main__':
    # this is run name 
    custom_run_name = "14features"
    x, y, feature_num = util.data_preprocessing(DATA_PATH+'/Exp_Sepsys_onset_14_Features_3hr.npy')
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, stratify = y, random_state=_SEED)
    X_train, X_test = util.convert_to_timeseries(X_train, X_test, 3, feature_num, scale=True)
    dynamic_selector = DynamicSelection(X_train, Y_train, X_test, Y_test, epoch=300, custom_filename=custom_run_name)
    dynamic_selector.run()