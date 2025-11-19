#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import numpy as np

class SimpleCNN(nn.Module):
    
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,3,padding = 'same')
        self.pool = nn.MaxPool2d(2, stride = 2)
        self.conv2 = nn.Conv2d(8, 16, 3,padding = 'same')
        self.conv3 = nn.Conv2d(16, 32, 3,padding = 'same')
        
        self.fc1 = nn.Linear(4*4*32, 2)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        
        
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.flatten(x,1)  
        x = self.fc1(x)
        x = F.softmax(x,dim = -1)
        return x

def splitData(X,Y,split = 0.8):
    nd = X.size(0)
    ids = torch.randperm(nd)
    ts = int(split*nd)
    X_train = X[ids[:ts],:,:,:]
    Y_train = Y[ids[:ts]]
    X_test = X[ids[ts:],:,:,:]
    Y_test = Y[ids[ts:]]
    return (X_train,X_test,Y_train,Y_test)


def trainSimpleCNN(X,Y):
    net = SimpleCNN()   
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9) 
    
    nrepochs = 100
    for epoch in range(nrepochs):  # loop over the dataset multiple times
        
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
    
        # print statistics
        print(loss.item())

    print('Finished Training')
    return net




def predictSimpleCNN(net,X):
    with torch.no_grad():
        output = net(X)
    _, Y = torch.max(output, 1)
     
    return Y




if __name__ == "__main__":
    # load data, change datadir path if your data is elsewhere
    datadir = 'HA3/inl3_to_students_python/'
    data = scipy.io.loadmat(datadir + 'FaceNonFace.mat')

    X = torch.tensor(data['X'].astype(int)).float()
    X = X.reshape(1,19,19,200)
    X = X.movedim((0,1,2,3),(1,2,3,0))
    Y = torch.tensor(data['Y'].astype(int))
    Y = (Y+1)//2 # change labels from (-1,1) to (0,1)
    Y = Y.flatten()


    X_train, X_test, Y_train, Y_test = splitData(X,Y,0.9)


    net = trainSimpleCNN(X_train,Y_train)

    Y2 = predictSimpleCNN(net,X_test)
    print(Y_test-Y2)

    nbr_trials = 100
    err_rates_train = np.zeros(nbr_trials)
    err_rates_test = np.zeros(nbr_trials)

    for i in range(nbr_trials):
        X_train, X_test, Y_train, Y_test = splitData(X,Y,0.8)

        net = trainSimpleCNN(X_train,Y_train)

        Y_pred_test = predictSimpleCNN(net, X_test)
        Y_pred_train = predictSimpleCNN(net, X_train)

        err_test = torch.count_nonzero(Y_pred_test - Y_test).item() / len(Y_test)
        err_train = torch.count_nonzero(Y_pred_train - Y_train).item() / len(Y_train)

        err_rates_test[i] = err_test
        err_rates_train[i] = err_train

    print("Mean test error CNN:", np.mean(err_rates_test))
    print("Mean train error CNN:", np.mean(err_rates_train))

