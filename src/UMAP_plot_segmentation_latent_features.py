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
from sklearn.manifold import TSNE   
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
import umap

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
fname = 'TTS_MI_plot0_umap'
target_names = ['TTS', 'MI']

#%%
def plot(x, colors):
    palette = np.array(sb.color_palette("hls", len(target_names)))  #Choosing color palette 
    # clist = []
    # for i in colors:
    #     clist.append(palette[i])
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(x)
    x = scaler.transform(x)
    # Create a scatter plot.
    f = plt.figure(figsize=(10, 10), dpi=600)
    ax = plt.subplot(aspect='equal')
    # ax.legend(['First line', 'Second line'])
    # ax.set(title="NCT-CRC-HE data T-SNE projection")
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=30, 
                    c=palette[colors])
    # Add the labels for each digit.
    txts = []
    # for i in range(len(target_names)):
    #     # Position of each label.
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), 
    #                           pe.Normal()])
    #     txts.append(txt)
        
    # ax.legend(target_names, ncols=len(target_names), 
    #           bbox_to_anchor=(0, 1),
    #           loc='best', fontsize='small')
    handles = [plt.Rectangle((0,0),1,1, color=palette[i]) for i in range(len(target_names))]
    plt.title("TTS and MI T-SNE projection", fontsize=20)
    plt.legend(handles, 
                target_names,
                loc='upper right',
                fontsize=15,
                ncols=len(target_names)
                )
    plt.xticks([]),plt.yticks([])
    plt.savefig(f'{fname}.pdf', bbox_inches='tight')
    return f, ax, txts


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
    X = np.concatenate((x_train,x_test))
    Y = np.concatenate((y_train, y_test)).astype(np.int8)
    
    # tsne_features = TSNE(perplexity=100).fit_transform(X) 
    # plot(tsne_features, Y)
    mapper = umap.UMAP().fit(X)
    # umap.plot.points(mapper, labels=Y)
    embedding = mapper.transform(X)
    plot(embedding, Y)
    
    break
