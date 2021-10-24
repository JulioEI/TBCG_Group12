# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 01:22:22 2021

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
#                            CNN HP EXPLORATION                               #
###############################################################################


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
import tensorflow as tf

import keras_tuner as kt

n_ch = data_Som2.shape[1]
input_shape = (int(fs*window_seconds),n_ch,1)



def model_builder(hp):
    
    model = Sequential()
    
    hp_filters1 = hp.Choice('filters_Conv1', [8,16,32])
    model.add(layers.Conv2D(filters = hp_filters1, kernel_size=(2,2), activation='relu',
                     input_shape=input_shape, padding='same', strides = (1,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    hp_filters2 = hp.Choice('filters_Conv2', [4, 8, 16])
    model.add(layers.Conv2D(filters = hp_filters2, kernel_size=(2,2), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    hp_filters3 = hp.Choice('filters_Conv3', [2, 4, 8])
    model.add(layers.Conv2D(filters = hp_filters3, kernel_size=(3,2), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    hp_filters4 = hp.Choice('filters_Conv4', [4, 8, 16])
    model.add(layers.Conv2D(filters = hp_filters4, kernel_size=(4,1), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    hp_filters4 = hp.Choice('filters_Conv4', [8, 16, 32])
    model.add(layers.Conv2D(filters = 16, kernel_size=(6,1), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    '''
    hp_filters5 = hp.Choicee('filters_Conv4', [4, 8, 16])
    model.add(layers.Conv2D(filters = 8, kernel_size=(6,1), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    '''
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.build()
    model.compile(
        optimizer= optimizers.Adam(learning_rate=1e-5), 
        loss='mean_absolute_error', 
        metrics=['mean_squared_error']  
    )

    return model

tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='kt_v1')


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x_train, y_train, epochs=10, validation_data = (x_validation, y_validation), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")