# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:00:54 2021

@author: Usuario
"""
import matplotlib.pyplot as plt
import numpy as np
import bcg_auxiliary as bcg

import sys
path_to_module = r"E:\\Users\Usuario\\Documents\\TheBrainCodeGame\\TBCG_SocioAstros\\"
sys.path.append(path_to_module)
import utils as ut

plt.plot(y_prediction_Dlx1)
plt.plot(y_validation_Dlx1)
plt.xlim([0,200])
plt.show()


th = 1e-1
y_prediction_Dlx1[y_prediction_Dlx1<th] = 0

def get_ripple_times_from_CNN_output(y_predicted, t_predicted, fs=1250, verbose = False,
                                     th_zero = 1e-1, th_dur = 0.4, th_exp = 1e-3):
    events = np.array([])
    window = 0
    while window < y_predicted.shape[0]:
        if y_predicted[window] <= th_zero: #if no ripple detected on this window jump to the next
            window += 1
        else: #ripple starts
            flag_dur = 0
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
                count = 1
                while ripple_end == 0:
                    if verbose:
                        print('\tripple still going on: ', window)
                    if y_predicted[window] <= th_exp:
                        if count>1:
                            en_pt = t_predicted[window-1, int(y_predicted[window-1]*t_predicted.shape[1]-1),:]
                        elif y_predicted[window-1]< th_dur:
                            flag_dur = 1
                            if verbose:
                                print('\tripple too short, discarding: ', y_predicted[window-1])
                        else:
                            st_pt = t_predicted[window-1, int(0.5*(1-y_predicted[window-1])*t_predicted.shape[1]),:]
                            en_pt = t_predicted[window-1, -int(0.5*(1-y_predicted[window-1])*t_predicted.shape[1]-1),:]
                            
                        ripple_end = 1 
                    if window+1==y_predicted.shape[0]: #last window
                         en_pt = t_predicted[window, int(y_predicted[window-1]*t_predicted.shape[1]-1), :]   
                         ripple_end = 1
                    window +=1
            if flag_dur == 0:
                if verbose: 
                    print("\tend of ripple: ", window-1, '(', en_pt[0]/fs, 's)')
                if events.shape[0]==0: #first ripple detected   
                    events = np.array([st_pt, en_pt]).T
                else:
                    events = np.vstack((events, np.array([st_pt[0], en_pt[0]])))
    return events



events_prediction_Dlx1= ut.get_ripple_times_from_CNN_output(y_prediction_Dlx1, indx_map_Dlx1, verbose=False)
events_validation_Dlx1= ut.get_ripple_times_from_CNN_output(y_validation_Dlx1, indx_map_Dlx1, verbose=False)

bcg.get_score(ripples_tags_Dlx1, events_validation_Dlx1, threshold=0.1)



events_prediction_Dlx1= ut.get_ripple_times_from_CNN_output(y_prediction_Dlx1, indx_map_Dlx1, 
                    th_zero = 5e-2, th_dur = 0.5, verbose=False)
