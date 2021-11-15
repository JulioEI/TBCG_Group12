# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 04:47:00 2021

@author: Julio
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
import tensorflow as tf

def model_builder_Unet(filters_Conv1 = 10, filters_Conv2 = 15, filters_Conv3=15, filters_Conv4 = 15,
                       input_shape = (50,8,1), learning_rate  = 1e-5):
    
    model = Sequential()
    model.add(layers.ZeroPadding2D(padding=((2,2),(0,0)), input_shape=input_shape))

    model.add(layers.Conv2D(filters = filters_Conv1, kernel_size=(5,5), activation='relu',
                      padding='valid', strides = (1,1)))
    model.add(layers.ReLU())
    model.add(layers.BatchNormalization())
    
    model.add(layers.ZeroPadding2D(padding=((2,2),(0,0))))

    model.add(layers.Conv2D(filters = filters_Conv2, kernel_size=(5,4), activation='relu',
                           padding='valid'))
    model.add(layers.ReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((input_shape[0],filters_Conv2)))
        
    
    model.add(layers.Conv1D(filters_Conv3,kernel_size =5, activation='relu', padding = 'valid', strides=5))
    model.add(layers.Conv1D(filters_Conv4,kernel_size =1, activation='relu', padding = 'valid', strides=1))
    model.add(layers.Conv1D(1,kernel_size =1, activation='relu', padding = 'valid', strides=1))
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
    return model



def model_builder_prob(filters_Conv1 = 32, filters_Conv2 = 16, filters_Conv3=8, filters_Conv4 = 16,
                  filters_Conv5 =16, filters_Conv6=8, input_shape = (50,8,1), 
                  learning_rate  = 1e-5):
    
    model = Sequential()
    model.add(layers.Conv2D(filters = filters_Conv1, kernel_size=(2,2), activation='relu',
                     input_shape=input_shape, padding='same', strides = (1,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters = filters_Conv2, kernel_size=(2,2), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters = filters_Conv3, kernel_size=(3,2), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters = filters_Conv4, kernel_size=(4,1), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(filters = filters_Conv5, kernel_size=(6,1), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(filters = filters_Conv6, kernel_size=(6,1), activation='relu',
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
    return model



def model_builder_binary(filters_Conv1 = 32, filters_Conv2 = 16, filters_Conv3=8, filters_Conv4 = 16,
                  filters_Conv5 =16, units_Dense = 60, input_shape = (50,8,1),
                  learning_rate = 1e-4):

    model = Sequential()
    
    model.add(layers.Conv2D(filters = filters_Conv1, kernel_size=(2,2), activation='relu',
                     input_shape=input_shape, padding='same', strides = (1,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(filters =filters_Conv2, kernel_size=(2,2), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(filters_Conv3, kernel_size=(3,2), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(filters_Conv4, kernel_size=(4,1), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2D(filters_Conv5, kernel_size=(6,1), activation='relu',
                           padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(units_Dense, activation='relu'))
    model.add(layers.Dense(50, activation='sigmoid'))
    
    model.build()
    model.compile(
        optimizer= optimizers.Adam(learning_rate=learning_rate), 
        loss='mean_absolute_error', 
        metrics=[tf.keras.metrics.TrueNegatives(thresholds=0.3), 
                 tf.keras.metrics.TruePositives(thresholds=0.3),
                 tf.keras.metrics.PrecisionAtRecall(0.5)]  
    )
    
    return model