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

DATA_DIR = '../new_data/EclinicalMedicine/'

folder_tts = sorted(os.listdir(DATA_DIR+'TTS'))
folder_mi = sorted(os.listdir(DATA_DIR+'STEMI'))


def reshape(a):
    return np.reshape(a, (a.shape[0], a.shape[1]*a.shape[2]*a.shape[3]*a.shape[4]))

x_tts = []
x_mi = []

for i in folder_tts:
    f = (np.load(DATA_DIR + 'TTS/'+i))
    x_tts.append(f)
    

for i in folder_mi:
    f = (np.load(DATA_DIR + 'STEMI/'+i))
    x_mi.append(f)


x_tts = np.stack(x_tts)
x_mi = np.stack(x_mi)
y_tts = np.ones(len(folder_tts))
y_mi = np.zeros(len(folder_mi))


#%%
fname = 'TTS_MI_plot_eclinical_medicine'
target_names = ['MI', 'TTS']

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

#%%
X = np.concatenate((x_tts,x_mi))
Y = np.concatenate((y_tts, y_mi)).astype(np.int8)

tsne_features = TSNE(perplexity=30).fit_transform(X) 
plot(tsne_features, Y)
