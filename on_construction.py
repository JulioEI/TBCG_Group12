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


datapath = "data/Som2"
data_Som2, fs, session_name_Som2 = bcg.load_data(datapath)
ripples_tags_Som2 = bcg.load_ripples_tags(datapath, fs)

datapath = "data/Amigo2"
data_Amigo2, fs, session_name_Amigo2 = bcg.load_data(datapath)
ripples_tags_Amigo2 = bcg.load_ripples_tags(datapath, fs)

n_ch = data_Som2.shape[1]
#downsample
desired_fs = 1250
down_sampling_factor =int(fs/desired_fs)

def downsample(array, factor):
    desired_length = int(factor*np.ceil(array.shape[0]/factor))
    padding_array = np.empty((desired_length-array.shape[0], array.shape[1]))
    padding_array[:] = np.nan
    array = np.vstack((array, padding_array))
    return np.nanmean(array.reshape(-1, factor, array.shape[1]),axis= 1)


data_Som2 = downsample(data_Som2, down_sampling_factor)
signal_Som2 = bcg.get_ripples_tags_as_signal(data_Som2, ripples_tags_Som2,desired_fs)

data_Amigo2 = downsample(data_Amigo2, down_sampling_factor)
signal_Amigo2 = bcg.get_ripples_tags_as_signal(data_Amigo2, ripples_tags_Amigo2, desired_fs)
fs = desired_fs



window_seconds = 0.04 #seconds
input_shape = (int(fs*window_seconds),8,1)


def window_stack(a, stepsize, width):
    n = a.shape[0]
    new_mat = np.zeros((np.floor(n/stepsize).astype(int), width, a.shape[1]),dtype=np.int16)
    ind = 0
    for window in range(new_mat.shape[0]-1):
        try:
            new_mat[window,:,:] = np.expand_dims(a[ind:ind+width,:], axis=0)
            ind = ind+stepsize
        except:
            print(ind, ind+width, a.shape[0])
            break
    return new_mat


overlapping = 0.6
x_train = np.expand_dims(window_stack(np.vstack((data_Som2,data_Amigo2)), 
                                      int((1-overlapping)*input_shape[0]), int(input_shape[0])),axis=3)

y_train = np.expand_dims(window_stack(np.vstack((np.expand_dims(signal_Som2,axis=1),
                                     np.expand_dims(signal_Amigo2,axis=1))),
                                     int((1-overlapping)*input_shape[0]), int(input_shape[0])),axis=3)


y_train = np.sum(y_train,axis =1)
y_train = np.squeeze((y_train>0.7*input_shape[0]).astype(int), axis=2)

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers



model = Sequential()
model.add(layers.Conv2D(filters = 4, kernel_size=(2,2), activation='relu',
                 input_shape=input_shape, padding='same', strides = (1,1)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters = 2, kernel_size=(2,2), activation='relu',
                       padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters = 4, kernel_size=(3,2), activation='relu',
                       padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters = 8, kernel_size=(4,1), activation='relu',
                       padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Conv2D(filters = 16, kernel_size=(6,1), activation='relu',
                       padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Conv2D(filters = 8, kernel_size=(6,1), activation='relu',
                       padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='relu'))

model.summary()
model.build()
model.compile(
    optimizer='adam', 
    loss='mean_absolute_error', 
    metrics=['mean_absolute_error']  
)

model.fit(x_train,y_train, shuffle = True, epochs = 1, batch_size = 32)

t = np.linspace(0, signal_Som2.shape[0]+signal_Amigo2.shape[0]-1, signal_Som2.shape[0]+signal_Amigo2.shape[0], dtype=int)
t_window = window_stack(np.expand_dims(t,axis=1), int((1-overlapping)*input_shape[0]), int(input_shape[0]))

y_predicted = model.predict(x_train[:150,:,:,:])
y_truth = y_train[:150,:]
t_predicted = t_window[:150,:,:]


events = ut.get_ripple_times_from_CNN_output(y_train[:2000,:], t_window[:2000], fs = fs, verbose = False)   
            



