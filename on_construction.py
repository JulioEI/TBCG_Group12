# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 01:30:05 2021

@author: Usuario
"""

import bcg_auxiliary as bcg
#import matplotlib.pyplot as plt
import numpy as np

import sys
path_to_module = r"E:\\Users\Usuario\\Documents\\TheBrainCodeGame\\TBCG_SocioAstros\\"
sys.path.append(path_to_module)
import utils as ut
import os 

fs=1250
window_seconds = 0.04 #seconds
overlapping = 0.6
batch_size = 32
learning_rate = 1e-5


###############################################################################
#                               TRAINING DATA                                 #
###############################################################################

### LOAD TRAIN DATA ###
datapath = "../data/Amigo2"
data_Amigo2, ripples_tags_Amigo2, signal_Amigo2, x_train_Amigo2, y_train_Amigo2, indx_map_Amigo2 = ut.load_data_pipeline(
    datapath, desired_fs=fs, window_seconds = window_seconds, overlapping = overlapping, zscore= True)

datapath = "../data/Som2"
data_Som2, ripples_tags_Som2, signal_Som2, x_train_Som2, y_train_Som2, indx_map_Som2 = ut.load_data_pipeline(
    datapath, desired_fs=fs, window_seconds = window_seconds, overlapping = overlapping, zscore=True)

### MERGE TRAIN DATA ###
x_train = np.vstack((x_train_Amigo2, x_train_Som2))
y_train = np.vstack((np.expand_dims(y_train_Amigo2,axis=1), np.expand_dims(y_train_Som2,axis=1)))
indx_map_train = np.vstack((indx_map_Amigo2, indx_map_Som2))

###############################################################################
#                             VALIDATION DATA                                 #
###############################################################################
### LOAD VALIDATION DATA ###
datapath = "../data/Dlx1"
data_Som2, ripples_tags_Dlx1, signal_Dlx1, x_validation_Dlx1, y_validation_Dlx1, indx_map_Dlx1 = ut.load_data_pipeline(
    datapath, desired_fs=fs, window_seconds = window_seconds, overlapping = overlapping, zscore=True)

datapath = "../data/Thy7"
data_Som2, ripples_tags_Thy7, signal_Thy7, x_validation_Thy7, y_validation_Thy7, indx_map_Thy7 = ut.load_data_pipeline(
    datapath, desired_fs=fs, window_seconds = window_seconds, overlapping = overlapping, zscore=True)

### MERGE VALIDATION DATA ###
x_validation = np.vstack((x_validation_Dlx1, x_validation_Thy7))
y_validation = np.vstack((np.expand_dims(y_validation_Dlx1,axis=1), np.expand_dims(y_validation_Thy7,axis=1)))
indx_map_validation = np.vstack((indx_map_Dlx1, indx_map_Thy7))

###############################################################################
#                            CNN ARCHITECTURE                                 #
###############################################################################
### CREATE CNN ARCHITECTURE ###
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
import tensorflow as tf

n_ch = data_Som2.shape[1]
input_shape = (int(fs*window_seconds),n_ch,1)

model = Sequential()
model.add(layers.Conv2D(filters = 12, kernel_size=(2,2), activation='relu',
                 input_shape=input_shape, padding='same', strides = (1,1)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters = 4, kernel_size=(2,2), activation='relu',
                       padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters = 4, kernel_size=(3,2), activation='relu',
                       padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters = 8, kernel_size=(4,1), activation='relu',
                       padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Conv2D(filters = 16, kernel_size=(6,1), activation='relu',
                       padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Conv2D(filters = 8, kernel_size=(6,1), activation='relu',
                       padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.build()
model.compile(
    optimizer= optimizers.Adam(learning_rate=learning_rate), 
    loss='mean_absolute_error', 
    metrics=['Accuracy']  
)

checkpoint_path = "training_v1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=10*batch_size)

###############################################################################
#                                  TRAIN CNN                                  #
###############################################################################

model.fit(x_train,y_train, shuffle = True, epochs = 5000, batch_size = batch_size, 
          callbacks=[cp_callback], validation_data = (x_validation, y_validation))
model.save_weights(checkpoint_path)





'''
y_predict = model.predict(x_train)
events_predicted = ut.get_ripple_times_from_CNN_output(y_predict, t_train, verbose=True)
events_truth = ut.get_ripple_times_from_CNN_output(y_train, t_train, verbose=True)
bcg.get_score (events_truth, events_predicted, threshold=0.1)
'''
