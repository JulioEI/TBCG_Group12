# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 20:55:01 2021

@author: Julio
"""
## CHANGE DIRECTORY TO 'code' FOLDER
import bcg_auxiliary as bcg
import os
import utils as ut
import tensorflow as tf

###############################################################################
#                              GENERAL PARAMETERS                             #
###############################################################################
fs=1250
window_seconds = 0.05 #seconds
overlapping = 0.7
binary = False
Unet = False
###############################################################################
#                                LOAD TEST DATA                               #
###############################################################################
### LOAD and PREPROCESS TEST DATA ###
datapath = "../data/test1"
data_test1, x_test1, indx_map_test1 = ut.load_test_data_pipeline(datapath, desired_fs=fs, 
                             window_seconds = window_seconds, overlapping = overlapping, zscore=True, binary = binary)

datapath = "../data/test2"
data_test2, x_test2, indx_map_test2 = ut.load_test_data_pipeline(datapath, desired_fs=fs, 
                             window_seconds = window_seconds, overlapping = overlapping, zscore=True, binary = binary)

###############################################################################
#                                LOAD MODEL                                   #
###############################################################################
#load model
model = tf.keras.models.load_model(os.path.join("/model/model_prob_vf.h5"))

###############################################################################
#                              GET PREDICTIONS                                #
###############################################################################
y_test1 = model.predict(x_test1)
y_test2 = model.predict(x_test2)

events_prediction_test1 = ut.get_ripple_times_from_CNN_output(y_test1, indx_map_test1,
                                              th_zero = 5e-1,th_dur = 0.01, verbose = False)
events_prediction_test2 = ut.get_ripple_times_from_CNN_output(y_test2, indx_map_test2,
                                              th_zero = 5e-1,th_dur = 0.01, verbose = False)
###############################################################################
#                               SAVE PREDICTIONS                              #
###############################################################################
bcg.write_results("../data/test1_results", 'test1',12, events_prediction_test1)
bcg.write_results("../data/test2_results", 'test2',12, events_prediction_test2)
