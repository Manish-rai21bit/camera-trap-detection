#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:24:13 2018

@author: manishrai
"""

import pandas as pd
import os, sys, random, ssl, csv
import urllib, urllib.request
# sys.path.join('/Users/manishrai/Desktop/UMN/Research/Zooniverse/Code/')

import pysftp #pip install sftp

hostname = "login.msi.umn.edu"
username = "xxxxxx"       
password = "xxxxxxxx"  

# read the csv
lst = list(pd.read_csv('../msi_image_names.csv'))
base_dir = "xxxx/xxxxx/xxxxx"

sftp = pysftp.Connection(hostname, username=username, password=password)

# This datasize is too much for my loacl.
# Splitting the downloading task into 5000 image batch
# 0:5000, 1:30 PM to 2:30 PM
# 5000: 10000 # 5000 images are taking up 4 GB so increasing the batch size
# 10000: 15000 # 7620 seconds
# 15000: 20000 # 5283 seconds
# 20000: 25000 # 4831 seconds
# 25000: end # 8055 seconds

import time
start_time = time.time()

for i, image in enumerate(lst[25000:]):
    sftp.get(base_dir + image + '.JPG')
    if i%500 == 0:
        print('downloaded {0} images'.format(i))
        print("---%s seconds ---" % (time.time() - start_time))
        
print("--- %s seconds ---" % (time.time() - start_time))
