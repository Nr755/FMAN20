#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""

import scipy
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def class_train(X, Y):
    # just store training data
    return (X, Y)

def classify(x, classification_data):
    X_train, Y_train = classification_data
    # compute squared Euclidean distances
    dists = np.sum((X_train - x) ** 2, axis=1)
    # index of closest training sample
    idx = np.argmin(dists)
    return Y_train[idx]

if __name__ == "__main__":
    # load data, change datadir path if your data is elsewhere
    datadir = 'HA3/inl3_to_students_python/'
    data = scipy.io.loadmat(datadir + 'FaceNonFace.mat')
    X = data['X'].transpose()
    Y = data['Y'].transpose()
    nbr_examples = np.size(Y,0)
    
    # This outer loop will run 100 times, so that you get a mean error for your
    # classifier (the error will become different each time due to the
    # randomness of train_test_split, which you may verify if you wish).
    nbr_trials = 100
    err_rates_test = np.zeros((nbr_trials,1));
    err_rates_train = np.zeros((nbr_trials,1));
    for i in range(nbr_trials):
        
        # First split data into training / testing (80% train, 20% test)
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
        nbr_train_examples = np.size(Y_train,0)
        nbr_test_examples = np.size(Y_test,0)
        
        # Now we can train our model!
        # YOU SHOULD IMPLEMENT THE FUNCTION class_train!
        classification_data = class_train(X_train, Y_train)
            
        # Next, let's use our trained model to classify the examples in the 
        # test data
        predictions_test = np.zeros((nbr_test_examples,1))
        for j in range(nbr_test_examples):
            # YOU SHOULD IMPLEMENT THE FUNCTION classify!
            predictions_test[j] = classify(X_test[j,:], classification_data)
        
       
        # We do the same thing again but this time for the training data itself!
        predictions_train = np.zeros((nbr_train_examples,1))
        for j in range(nbr_train_examples):
            # YOU SHOULD IMPLEMENT THE FUNCTION classify!
            predictions_train[j] = classify(X_train[j,:], classification_data)
            
    
    
        # We can now proceed to computing the respective error rates.
        pred_test_diff = predictions_test - Y_test
        pred_train_diff = predictions_train - Y_train
        err_rate_test = np.count_nonzero(pred_test_diff) / nbr_test_examples
        err_rate_train = np.count_nonzero(pred_train_diff) / nbr_train_examples
        
        # Store them in the containers
        err_rates_test[i] = err_rate_test
        err_rates_train[i] = err_rate_train
    print(np.mean(err_rates_test))
    print(np.mean(err_rates_train))
    
        
# pick one index from test set where label = 1 (face) and one where label = -1 (non-face)
face_idx = np.where(Y_test == 1)[0][0]      # first face in test set
nonface_idx = np.where(Y_test == -1)[0][0]  # first non-face in test set

# extract the vectors and reshape to 19x19
face_img = X_test[face_idx, :].reshape(19, 19)
nonface_img = X_test[nonface_idx, :].reshape(19, 19)

# classify them with your classifier
pred_face = classify(X_test[face_idx, :], classification_data)
pred_nonface = classify(X_test[nonface_idx, :], classification_data)

# plot the results
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(face_img, cmap='gray')
plt.title(f"Face (true=1, pred={pred_face})")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(nonface_img, cmap='gray')
plt.title(f"Non-Face (true=-1, pred={pred_nonface})")
plt.axis('off')

plt.show()



