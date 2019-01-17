#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:53:44 2019

@author: manishrai
"""

import os, sys, csv
import tensorflow as tf
import dataset_util
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""This function reads a raw image, resizes it with aspect ratio preservation and returns the byte string"""
from PIL import Image
import numpy as np
import io



def resize_jpeg(image,  max_side):
    """ Take Raw JPEG resize with aspect ratio preservation
         and return bytes
    """
    img = Image.open(image)
    img.thumbnail([max_side, max_side], Image.ANTIALIAS)
    b = io.BytesIO()
    img.save(b, 'JPEG')
    image_bytes = b.getvalue()
    return image_bytes


""" This function creates a tfrecord example from the dictionary element!"""
def create_tf_example(data_dict):
    #with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
    encoded_jpg = resize_jpeg(os.path.join('/panfs/roc/groups/5/packerc/shared/albums/SER/', data_dict) + '.JPG',  1000)
    #encoded_jpg_io = io.BytesIO(encoded_jpg)
    #image = Image.open(encoded_jpg_io)
    #width, height = image.size
    filename = data_dict.encode('utf-8')
    image_format = b'jpg'
   
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format)
    }))

    
    return tf_example

"""This iterates over each dictionary item, creates tf examples, 
    serializes the tfrecord examples and writes to a tfrecord file!!!
    As of now, it saves the TFRecord file in the home directory where the code is executed"""
def encode_to_tfr_record(test_feature, out_tfr_file):
    with tf.python_io.TFRecordWriter(out_tfr_file) as writer:
        count = 0
        for v in test_feature:
            count+=1
            if count%500==0:
                print("processing event number %s : %s" % (count, v))
            example = create_tf_example(v)
            writer.write(example.SerializeToString())


def main():
    with open("/home/packerc/rai00007/camera-trap-detection/data/LILA/msi_snapshot_serengeti.csv",'r') as f:
        l = []
        rd = csv.reader(f)
        for val in rd:
            l.append(val)
       
    event_dict = l[0]
    event_dict1 = event_dict[20000:50000]
    encode_to_tfr_record(event_dict1, 'snapshot_serengeti_s01_s06-20000-50000.record')

if  __name__=='__main__':
    # event_dict = ['S1/B04/B04_R1/S1_B04_R1_PICT0040']
    # encode_to_tfr_record(event_dict, 'snapshot_serengeti_s01_s06.record')
    main()
