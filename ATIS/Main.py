#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:48:55 2019

@author: giulia
"""



from functions import*

#ATIS parameters
ATIS_width=304
ATIS_height=240


#ATISconvert_data: convert data.log coming from the zynq to a txt file (data.log.txt)
dataset_path = '/home/adampcloth/PycharmProjects/SpiNNiCub/ATIS/data_surprise'
ATISconvert_data(ATIS_width, ATIS_height, dataset_path)

#Loading ATIS dataset: event= (timestamp, channel, y, x, polarity) for the left/right channel (left/right camera)
[left_data,right_data]=ATISload_data(ATIS_width, ATIS_height, dataset_path)
