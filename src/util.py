import os
import random
import numpy as np
import gc
import pickle
import matplotlib.pyplot as plt 

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score

def set_seed(_SEED):
    os.environ['PYTHONHASHSEED']=str(_SEED)
    np.random.seed(_SEED)  # Numpy module.
    random.seed(_SEED)  # Python random module.
    
    torch.manual_seed(_SEED)
    torch.cuda.manual_seed(_SEED)
    torch.cuda.manual_seed_all(_SEED)  # if you are using multi-GPU.
   
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

_SEED = 42
set_seed(_SEED)

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Average Reward')       
    plt.xlabel('Epochs')
    plt.plot(x, running_avg, color='#a44027')
    plt.savefig(filename)

def remove_diff(data):
    i=0
    _arr=[]
    while (i<data.shape[1]):
        if (i%5 == 0):
            j=0
            while(j<3 and i+j<data.shape[1]-2):
                _arr.append(i+j)
                j = j+1
        i=i+1
    data = data[:,_arr]
    return data

def data_preprocessing(filename):
    data = np.load(filename)
    
    filename = filename.split("_")
    feature_num = [int(filename[x]) for x in range(len(filename)) if filename[x].isdigit()]
    
    data_null = data[(data[:,[x for x in range(1,data.shape[1])]]==0).all(1)]
    print("Removed data points that are zero vectors: "+ str(len(data_null)))
    data = data[~(data[:,[x for x in range(1,data.shape[1])]]==0).all(1)]
    
    train_x = data[:,[x for x in range(1, data.shape[1])]]
    train_y = data[:,[0]]
    train_y = train_y.reshape(1,-1)[0]
    
    train_x = remove_diff(train_x)
    return train_x, train_y, feature_num[0]

def stratify_2_class(train_x, train_y):
    #concatenate train_x and train_y and shuffle them
    train_xy = np.concatenate((train_x, train_y.reshape(-1,1)), axis = 1)
    
    y_index = np.shape(train_xy)[1]
    # sort based on class
    train_xy = sorted(train_xy, key=lambda x: x[y_index-1], reverse=True)

    # get total minority class    
    _sum = 0
    for i in train_xy:
        _sum = _sum + i[y_index-1]
    _sum = int(_sum)

    # get ratio
    ratio = np.round((len(train_xy)-_sum)/_sum)

    train_xy_1 = train_xy[0:_sum]
    train_xy_0 = train_xy[_sum:len(train_xy)]

    train_xy_new = []
    k_0 = 0
    k_1 = 0
    for i in range(len(train_xy)):
        if i % ratio == 0 and k_1<len(train_xy_1):
            train_xy_new.append(train_xy_1[k_1])
            k_1 = k_1 + 1
        else:
            train_xy_new.append(train_xy_0[k_0])
            k_0 = k_0 + 1
        
    train_xy_new = np.array(train_xy_new)
    
    #     split matrix to x and y
    train_x = train_xy_new[:, [x for x in range(np.shape(train_xy_new)[1]-1)]]
    train_y = train_xy_new[:, train_x.shape[1]].reshape(np.shape(train_y)[0], 1).astype(int)
    
    return train_x, train_y

def convert_to_timeseries(X_train, X_test, time_step, feature_num, scale=True):
    
    if scale == True:
        transformer = MaxAbsScaler().fit(X_train)
        X_train = transformer.transform(X_train)
        X_test = transformer.transform(X_test)

    X_train = X_train.reshape(X_train.shape[0],feature_num, time_step)
    X_train = np.array([X_train[x].T for x in range(X_train.shape[0])])
    X_test = X_test.reshape(X_test.shape[0],feature_num, time_step)
    X_test = np.array([X_test[x].T for x in range(X_test.shape[0])])

    return X_train, X_test