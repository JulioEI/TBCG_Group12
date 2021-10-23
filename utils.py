# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 14:54:31 2021

@author: Julio
"""
import numpy as np



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
                        en_pt = t_predicted[window-1, int(y_predicted[window-1]*t_predicted.shape[1]),:]
                        ripple_end = 1 
                    if window+1==y_predicted.shape[0]: #last window
                         en_pt = t_predicted[window, int(y_predicted[window-1]*t_predicted.shape[1]), :]   
                         ripple_end = 1
                    window +=1
                if verbose: 
                    print("\tend of ripple: ", window-1, '(', en_pt[0]/fs, 's)')

            if events.shape[0]==0: #first ripple detected
                events = np.array([st_pt, en_pt]).T
            else:
                events = np.vstack((events, np.array([st_pt[0], en_pt[0]])))
                
    return events