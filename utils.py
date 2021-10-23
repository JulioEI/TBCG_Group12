# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 14:54:31 2021

@author: Julio
"""
import numpy as np

def mov_av_downsample(array, win):
    desired_length = int(win*np.ceil(array.shape[0]/win))
    array = np.pad(array.astype(float), ((0, desired_length-array.shape[0]), (0, 0)), 
                  mode='constant', constant_values=np.nan)
    return np.nanmean(array.reshape(-1, win, array.shape[1]),axis= 1)


def window_stack(a, stepsize, width):
    n_t = a.shape[0]
    if np.ndim(a)==1:
        a = np.expand_dims(a, axis=1)
    n_ch = a.shape[1]
    new_mat = np.zeros((np.ceil((n_t-width)/stepsize).astype(int)+1, width, n_ch),dtype=np.int64)
    ind = 0
    for window in range(new_mat.shape[0]):
        if ind+width>n_t:
            ind = n_t-width
            print(ind, ind+width, a.shape[0])
        new_mat[window,:,:] = np.expand_dims(a[ind:ind+width,:], axis=0)
        ind = ind+stepsize
    return new_mat


def adapt_input_to_CNN(array, window_size, overlapping):
    indx_map = np.linspace(0, array.shape[0]-1, array.shape[0], dtype=int)
    array_reshape = np.expand_dims(window_stack(array, int((1-overlapping)*window_size),
                                                    window_size), axis=3)
    indx_map = window_stack(indx_map, int((1-overlapping)*window_size),
                                                    window_size)
    return array_reshape, indx_map


def adapt_label_to_CNN(array, window_size, overlapping):
    label = window_stack(array,int((1-overlapping)*window_size), window_size)
    return np.squeeze(np.sum(label, axis=1)/window_size)


def get_ripple_times_from_CNN_output(y_predicted, t_predicted, fs=1250, verbose = False):
    events = np.array([])
    window = 0
    while window < y_predicted.shape[0]:
        if y_predicted[window] == 0: #if no ripple detected on this window jump to the next
            window += 1
        else: #ripple starts
            st_pt = t_predicted[window,int(-y_predicted[window]*t_predicted.shape[1]),:]
            if verbose:
                print('\nStart ripple: ', window, '(', st_pt[0]/fs, 's)')
    
            if window+1>=y_predicted.shape[0]: #last window then ripple ends 
                en_pt = t_predicted[window, -1:, :]
                window+1
            else: #start looking into future windows to find the end of the ripple
                if verbose:
                    print('Computing end of ripple: ')
                ripple_end = 0
                window+=1
                while ripple_end == 0:
                    if verbose:
                        print('\tripple still going on: ', window)
                    if y_predicted[window] == 0:
                        en_pt = t_predicted[window-1, int(y_predicted[window-1]*t_predicted.shape[1]-1),:]
                        ripple_end = 1 
                    if window+1==y_predicted.shape[0]: #last window
                         en_pt = t_predicted[window, int(y_predicted[window-1]*t_predicted.shape[1]-1), :]   
                         ripple_end = 1
                    window +=1
                if verbose: 
                    print("\tend of ripple: ", window-1, '(', en_pt[0]/fs, 's)')

            if events.shape[0]==0: #first ripple detected
                events = np.array([st_pt, en_pt]).T
            else:
                events = np.vstack((events, np.array([st_pt[0], en_pt[0]])))
                
    return events


