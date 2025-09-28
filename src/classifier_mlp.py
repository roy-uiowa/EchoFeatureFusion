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
from sklearn.neural_network import MLPClassifier
import cv2
import tensorflow as tf
from tensorflow.keras import models, layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam

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
def regnet(input_shape):
    inputs = layers.Input(input_shape, dtype=tf.float32)
    dense1 = layers.Dense(64, activation='relu')(inputs)
    dense1 = layers.Dropout(0.25)(dense1)
    dense2 = layers.Dense(256, activation='relu')(dense1)
    dense2 = layers.Dropout(0.25)(dense2)
    dense3 = layers.Dense(512, activation='relu')(dense2)
    dense3 = layers.Dropout(0.25)(dense3)
    output = layers.Dense(2, activation='softmax')(dense3)
    model = models.Model(inputs, output, name="RegNet")
    return model

#%%
accuracy = []
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
    final_index= feature_selection(x_train,y_train)
    print(final_index.shape)
    ft_x_train = x_train[:, final_index]
    ft_x_train = np.squeeze(ft_x_train)
    ft_x_test = x_test[:, final_index]
    ft_x_test = np.squeeze(ft_x_test)
    #%%
    # clf = MLPClassifier(random_state=0,
    #                       max_iter=1000,).fit(ft_x_train, y_train)
    # test_acc = clf.score(ft_x_test, y_test)
    # print('Accuracy:', test_acc)
    # accuracy.append(test_acc)
    #%%
    # ft_map = create_feature_map(final_index, x_test_mi)
    
    #%%
    
    filepath = str('TTS_MI_regnet_dense_fold_{:02d}.hdf5'.format(i+1))
    y_train = tf.keras.utils.to_categorical(y_train, 2)
    y_test = tf.keras.utils.to_categorical(y_test, 2)
    
    TRAIN = False
    LRATE = 0.000001
    EPOCHS = 1000
    BATCH_SIZE = 64
    num_labels = 2
    n_train, n_features = ft_x_train.shape
    input_shape = (n_features)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LRATE,
        decay_steps=2500,
        decay_rate=0.8,
        staircase=False)
    reg_model = regnet(input_shape)
    reg_model.compile(optimizer=Adam(learning_rate=lr_schedule), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

    if TRAIN:
        tf.keras.backend.clear_session()
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max')
        stop = EarlyStopping(monitor='val_accuracy', patience=200, mode="max")
        callbacks_list = [checkpoint, stop]
        regnet_history = reg_model.fit(ft_x_train, y_train,
                                       verbose=1,
                                       batch_size = BATCH_SIZE,
                                       validation_data=(ft_x_test, y_test),
                                       shuffle=True,
                                       callbacks=callbacks_list, epochs=EPOCHS)
    else:        
        tf.keras.backend.clear_session()
        reg_model.load_weights(filepath)
        _, acc = reg_model.evaluate(ft_x_test,  y_test, batch_size=4, verbose=1)
        pred_prob = reg_model.predict(ft_x_test, batch_size=4)
        accuracy.append(acc)


avg = sum(accuracy) / len(accuracy) 
print('Average_accuracy = ', avg)
