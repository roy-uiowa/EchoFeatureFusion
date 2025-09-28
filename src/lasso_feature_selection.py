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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

DATA_DIR = '../data/features/'

folds = sorted(os.listdir(DATA_DIR))

def reshape(a):
    return np.reshape(a, (a.shape[0], a.shape[1]*a.shape[2]*a.shape[3]*a.shape[4]))
#%%
from sklearn.linear_model import Lasso
def feature_selection(data,survival_info, alpha):
    clf = Lasso(alpha=alpha)
    clf.fit(data,survival_info)
    idx = np.nonzero(clf.coef_)
    idx = np.array(idx)
    return idx

#%%

alpha_value = np.concatenate((np.arange(0.1, 0.01, -0.001),
                              np.arange(0.01, 0, -0.00001),
                              ))

# alpha_value = np.arange(0.1, 0.01, -0.001)
mean_accuracy = []
selected_features = []
effective_alpha = []

for alpha in alpha_value:
    accuracy = []
    feature_size = []
    for i in range(4):
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
        # min_max_scaler = MinMaxScaler()
        
        # x_train = min_max_scaler.fit_transform(x_train)
        # x_test = min_max_scaler.fit_transform(x_test)
        #%%
        final_index= feature_selection(x_train,y_train, alpha)
        feature_size.append(final_index.shape[1])
        # print(final_index.shape)
        x_train = x_train[:, final_index]
        x_train = np.squeeze(x_train)
        x_test = x_test[:, final_index]
        x_test = np.squeeze(x_test)
        #%%
        clf = LogisticRegression(random_state=0,
                              max_iter=1000,).fit(x_train, y_train)
        test_acc = clf.score(x_test, y_test)
        
        # print('Accuracy:', test_acc)
        accuracy.append(test_acc)
        # break
    avg = sum(accuracy) / len(accuracy) 
    mean_accuracy.append(float("{:.4f}".format(avg)))
    avg_features = sum(feature_size) / len(feature_size) 
    selected_features.append(round(avg_features))
    print('Average_accuracy = ', avg)
#%%
max_acc = max(mean_accuracy)
max_idx = mean_accuracy.index(max_acc)
max_at_alpha = alpha_value[max_idx]
max_at_features = selected_features[max_idx]
print(max_at_alpha, max_at_features, max_acc)

merged_list = tuple(zip(mean_accuracy, selected_features))
seen = set()
filtered_merged_list = [(a, b) for a, b in merged_list
               if not (a in seen or seen.add(a))]

# mean_accuracy, selected_features = map(None, *filtered_merged_list)
mean_accuracy, selected_features = zip(*filtered_merged_list)

from matplotlib import pyplot as plt    
# import seaborn as sns
plt.style.use('seaborn')
fig = plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(x=selected_features, y=mean_accuracy,color='purple', marker='x',alpha=0.5)
plt.rc('font', size=15) 
plt.rc('axes', labelsize=15)
plt.rcParams.update({'font.size': 15})
# single vline with full ymin and ymax
plt.vlines(x=max_at_features, ymin=0.7, ymax=max_acc, colors='green', ls=':', lw=2, label='vline_single - full height')
plt.hlines(y=max_acc, xmin=0, xmax=max_at_features, colors='green', ls=':', lw=2, label='vline_single - full height')
plt.text(max_at_features, max_acc, f'({max_at_alpha}, {max_at_features}, {max_acc})',color='green') 
ax = plt.gca()
ax.set_xlim([-10, 500])
# ax.set_ylim([0.7, 1])
ax.set_xlabel('Number of selected features')
ax.set_ylabel('Average CV accuracy')
fig.savefig('lasso_feature_selection.pdf',
            bbox_inches = 'tight',)
