#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:23:22 2023

@author: troy3
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:38:33 2023

@author: troy3
"""

import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import cv2
from matplotlib import pyplot as plt

DATA_DIR = '../data/features/'

folds = sorted(os.listdir(DATA_DIR))

def reshape(a):
    return np.reshape(a, (a.shape[0], a.shape[1]*a.shape[2]*a.shape[3]*a.shape[4]))
#%%
from sklearn.linear_model import Lasso
def feature_selection(data,survival_info, alpha=0.006):
    clf = Lasso(alpha=alpha)
    clf.fit(data,survival_info)
    idx = np.nonzero(clf.coef_)
    idx = np.array(idx)
    return idx
#%%    
def create_feature_map(final_index, x_train):
    final_index = np.squeeze(final_index)
    mask = np.zeros_like(x_train)
    for i in range(mask.shape[0]):
        row = mask[i]
        row[final_index] = 1
        mask[i] = row
    x_train = np.multiply(x_train,mask)
    # x_train = mask.copy()
    x_train_m = np.mean(np.reshape(x_train, (x_train.shape[0],2,7,8,32)), axis=(1,-1))
    x_train_res = []
    for n in range(x_train.shape[0]):
        intrp = cv2.resize(x_train_m[n], (128, 112))
        intrp = intrp - np.min(intrp)
        intrp = intrp / np.max(intrp)
        x_train_res.append(intrp)
    x_train_res = np.array(x_train_res).astype(np.float32)
    print(x_train.shape)
    return x_train_res
#%%
alpha_value = np.concatenate((np.arange(0.1, 0.01, -0.001),
                              np.arange(0.01, 0, -0.00001),
                              ))
accuracy = []
for alpha in alpha_value:
    i = 1
    fold_dir = f'{DATA_DIR}Fold_0{i+1}/' 
    # print(os.listdir(fold_dir))
    
    x_train_tts = reshape(np.load(fold_dir + f'Train_features_TTS_fold_0{i+1}.npy'))
    y_train_tts = np.ones(x_train_tts.shape[0])
    x_train_mi = reshape(np.load(fold_dir + f'Train_features_MI_fold_0{i+1}.npy'))
    y_train_mi = np.zeros(x_train_mi.shape[0])
    
    x_test_tts = reshape(np.load(fold_dir + f'Test_features_TTS_fold_0{i+1}.npy'))
    y_test_tts = np.ones(x_test_tts.shape[0])
    x_test_mi = reshape(np.load(fold_dir + f'Test_features_MI_fold_0{i+1}.npy'))
    y_test_mi = np.zeros(x_test_mi.shape[0])
    
    # print(x_train_tts.shape, y_train_tts.shape)
    # print(x_train_mi.shape, y_train_mi.shape)
    # print(x_test_tts.shape, y_test_tts.shape)
    # print(x_test_mi.shape, y_test_mi.shape)
    
    x_train = np.concatenate((x_train_tts, x_train_mi))
    y_train = np.concatenate((y_train_tts, y_train_mi))
    x_test = np.concatenate((x_test_tts, x_test_mi))
    y_test = np.concatenate((y_test_tts, y_test_mi))
    
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    #%%
    final_index= feature_selection(x_train,y_train, alpha)
    print(final_index.shape)
    ft_x_train = x_train[:, final_index]
    ft_x_train = np.squeeze(ft_x_train)
    ft_x_test = x_test[:, final_index]
    ft_x_test = np.squeeze(ft_x_test)
    #%%
    # clf = LogisticRegression(random_state=0,
    #                       max_iter=10000,).fit(ft_x_train, y_train)
    
    clf = RandomForestClassifier(max_depth=20, random_state=0).fit(ft_x_train, y_train)
    test_acc = clf.score(ft_x_test, y_test)
    print('Accuracy:', test_acc)
    accuracy.append(test_acc)
    #%%
    # ft_map = create_feature_map(final_index, x_test_mi)
    
max_acc = max(accuracy)
max_idx = accuracy.index(max_acc)
max_at_alpha = alpha_value[max_idx]
print(max_at_alpha, max_acc)

plt.scatter(range(len(accuracy)), accuracy)
