"""Set of functions for creating TFRecord files with box leval annotations.
Contains TFRecord encoder and decored and supporting functions for dealing
with TFRecord."""

import tensorflow as tf
import os, csv, io
from PIL import Image

from data.utils import dataset_util
import data_prep.data_prep_utils as dataprep_utils
import data.image as img


"""This function takes the GSSS bounding box dataset published by in the paper
    and converts it into a dictionary object. The column names in the CSV has to maintained.
    
1. csvtodict - creates dictionary objects. have to rename this function.
2. dicttojson - JSON dump of the dictionary.
3. jsontodict - JSON file to a dictionary for reading into a TFRecord.
4. create_tf_example - Creates a tf_example.
5. encode_to_tfr_record - Creates a TF Record file"""
def csvtodict(image_filepath, bb_data):
    lst = []
    record_dict = {}
    csvfile = open(os.path.join(bb_data), 'r')
    csvdata = csv.reader(csvfile, delimiter=',')
    first_row = next(csvdata)
    for row in csvdata:
        if row[0] not in record_dict: # the condition in lst2 is to pick only the images usd by schneider
            record_dict[row[0]] = {'metadata' : {"SiteID": row[0].split('/')[1],
                                  "DateTime": "placeholder", 
                                  "Season": row[0].split('/')[0]},
                                    'images' : [{"Path" : os.path.join(image_filepath, row[0] + '.JPG'), #points to the route of image on the disk
                                "URL" : 'placeholder',
                                "dim_x" : 'placeholder',
                                "dim_y" : 'placeholder',
                                "image_label" : "tbd", # This is the primary label in case we want to have some for the whole image
                                'observations' : []
                               }]
                                    }
        record_dict[row[0]]['images'][0]['observations'].append({'bb_ymin': row[3], 
                                                   'bb_ymax': row[5], 
                                                      'bb_primary_label': row[1], 
                                                      'bb_xmin': row[2], 
                                                      'bb_xmax': row[4], 
                                                      'bb_label': {"species" : row[1],
                                                    "pose" : "standing/ sitting/ running"
                                                }})
    return record_dict


""" This function creates a tfrecord example from the dictionary element!"""
def create_tf_example(data_dict, 
                      label_map
                     ):
    encoded_jpg = img.resize_jpeg((data_dict['images'][0]['Path']),  1000)
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    width = int(width)
    height = int(height)

    filename = data_dict['images'][0]['Path'].encode('utf-8')
    image_format = b'jpg'
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for bb_record in data_dict['images'][0]['observations']:
        xmins.append(float(bb_record['bb_xmin']))
        xmaxs.append(float(bb_record['bb_xmax']))
        ymins.append(float(bb_record['bb_ymin']))
        ymaxs.append(float(bb_record['bb_ymax']))
        classes_text.append(bb_record['bb_primary_label'].encode('utf8'))
        classes.append(label_map[bb_record['bb_primary_label']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    
    return tf_example

"""This iterates over each dictionary item, creates tf examples, 
    serializes the tfrecord examples and writes to a tfrecord file!!!
    As of now, it saves the TFRecord file in the home directory where the code is executed"""
def encode_to_tfr_record(bounding_box_dict, label_map, out_tfr_file):
    with tf.python_io.TFRecordWriter(out_tfr_file) as writer:
        count = 0
        for k, v in bounding_box_dict.items():
            count+=1
            if count%500==0:
                print("processing event number %s : %s" % (count, k))
            example = create_tf_example(v, label_map)
            writer.write(example.SerializeToString())
            
