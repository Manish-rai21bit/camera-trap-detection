#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 15:37:10 2018

@author: manishrai
"""
import json, os, csv, random
import get_images_from_url
# import numpy as np
import pandas as pd
# import tensorflow as tf

filepath = '/Users/manishrai/Desktop/UMN/Research/Zooniverse/Data/SnapshotSafari/snapshot_serengeti/db'
# Read json blob
def read_json(json_file):
    with open(json_file) as f:
        json_dict = json.load(f)
    return json_dict

# Summarizing the data in the json_dict get idea of split criteria 
def get_summary_from_dict(json_dict):
    species_cnt_dict = {}
    i = 0
    for k, v in json_dict.items():
        # summarizing species count
        if v['label'] not in species_cnt_dict:
            species_cnt_dict[v['label']] = 1
        else:
            species_cnt_dict[v['label']] += 1             
        # summarizing some other field
        i += 1
        print("processing dict element: ", i) if i%10000 == 0 else ""
    return species_cnt_dict

# This reads the json file into the json_file object
"""This dictionary object looks like this:
 {'fnames': ['ASG000a0sz_2.jpeg', 'ASG000a0sz_1.jpeg', 'ASG000a0sz_0.jpeg'],
 'file_paths': [None, None, None],
 'label': 'wildebeest',
 'urls': ['http://www.snapshotserengeti.org/subjects/standard/50c219208a607540b9072293_0.jpg',
  'http://www.snapshotserengeti.org/subjects/standard/50c219208a607540b9072293_1.jpg',
  'http://www.snapshotserengeti.org/subjects/standard/50c219208a607540b9072293_2.jpg'],
 'meta_data': {'Moving': '0.545454545455', 'Standing': '0.636363636364', 'LocationX': '710006',
  'NumBlanks': '1', 'Babies': '0', 'Interacting': '0', 'Evenness': '0', 'Species': 'wildebeest',
  'Eating': '0.454545454545', 'NumImages': '3', 'Resting': '0', 'NumSpecies': '1', 'LocationY': '9716196',
  'SiteID': 'I10', 'DateTime': '2011-11-14 06:19:58', 'Count': '11-50', 'CaptureEventID': 'ASG000a0sz',
  'NumVotes': '11',
  'NumClassifications': '12'}, 'img_ids': ['ASG000a0sz_2', 'ASG000a0sz_1', 'ASG000a0sz_0']}   
"""

json_file = read_json(os.path.join(filepath, 'subject_set.json'))

# Once the json is loaded we select the events with one animal and write it to 
# a csv. This is a much smaller csv to can be loaded using pandas:
with open(os.path.join(filepath, 'subject_set_count1.csv'), 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerow(['CaptureEventID', 'Species', 'DateTime', 'SiteID', 'NumSpecies', 'Count', 'urls'])
    for k, v in json_file.items():
        if v['label']!='blank'  and v['meta_data']['Count']=='1':
            # for i in len(v['url']):
            csv_writer.writerow([k, 
                                v['meta_data']['Species'], 
                                v['meta_data']['DateTime'],
                                v['meta_data']['SiteID'],
                                v['meta_data']['NumSpecies'],
                                v['meta_data']['Count'], 
                                v['urls'][0]] # Choosing the first image. No rational behind it. 
                               )
            
# loading the csv to a pandas dataframe. will be easier for visualization and 
# minor analysis
df1 = pd.read_csv(os.path.join(filepath, 'subject_set_count1.csv'))

# Generating statistics on the distribution of animals in the images. This will
# help in getting even samples from the whole pool
df2 = df1[['Species', 'Count']].groupby(by=['Species'], axis=0, as_index=False).count()
df2 = df2.sort_values(by = 'Count')
df2['pct_cnt'] = round(df2['Count']*100/209180, 1)


df2.head()
df1.shape == df1.drop_duplicates(keep='first').shape # if True then no duplicates present

# sampling  function
def get_sampled_subject_set(data_df, Species, species_count):
    if species_count <= 100:
        return list(data_df[data_df['Species']==Species]['CaptureEventID'])
    elif species_count > 100 and species_count <= 1000:
        return list(random.sample(list(df1[df1['Species']==Species]['CaptureEventID']), 100))
    else:
        return list(random.sample(list(df1[df1['Species']==Species]['CaptureEventID']), 500))


# sorted(species_cnt_dict.items(), key=lambda kv: kv[1], reverse = True)
        
    
# Creating a list of event ids to be used for the subject set
event_ids = []
for s in df2['Species']:
    event_ids = event_ids + get_sampled_subject_set(df1, s, int(df2[df2['Species']==s]['Count']))

df3 = df1[df1['CaptureEventID'].isin(event_ids)]
outpath = '/Users/manishrai/Desktop/test_dir/'

get_images_from_url(df3, image_name_index=0, url_col_index=6, outpath = '/Users/manishrai/Desktop/test_dir/')

# once we have the subject set id's we can use them to download the real images 
# We might need some joins with the all_images dataset. But, for this case,
# I will extract the URLs from the json