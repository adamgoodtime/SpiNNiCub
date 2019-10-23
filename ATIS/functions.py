#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:54:08 2019

@author: giulia
"""

import numpy as np

def ATISconvert_data(ATIS_width, ATIS_height, path):
    
    from decode_events import*
    
    #Assuming channel 0 as left camera and channel 1 as right camera
    # Import data.log and Decode events
    dm = DataManager()
    dm.load_AE_from_yarp(path)
    print('ATIS data processing ended')
    
def ATISload_data(ATIS_width, ATIS_height, path):
    # Loading decoded events; data(timestamp, channel, x, y, polarity)  
    stereo_data=np.loadtxt(path+'/decoded_events.txt', delimiter=',' ) 
    [left_data,right_data]=split_stereo_data(stereo_data)

    
    return left_data,right_data

def split_stereo_data(stereo_data):
    width_stereo_data, height_stereo_data=stereo_data.shape
    print('Loading ATIS data ended')
    
    left_data=[]
    right_data=[]
    
    for i in range(1,width_stereo_data):    
        if stereo_data[i,1]==0:
            left_data.append(stereo_data[i,:])       
        else:
            right_data.append(stereo_data[i,:]) 
    return left_data, right_data