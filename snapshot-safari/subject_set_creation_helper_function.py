#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:20:34 2018

@author: manishrai
"""

import urllib, urllib.request
import os, ssl
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from run_inference_graph import load_image_into_numpy_array, run_inference_for_single_image

"""The set of functions here will help in creating the subject set. 
Subject set creation for bounding box requires the following functions:
    get_images_from_url - Download the images from the url's given into a local path
    run_inference_graph_images on the downloaded images"""

def get_images_from_url(dataset, image_name_index, url_col_index, outpath):
    """This function takes in a dataframe and downloads the images from the url's in the column.
    arguments:
        dataset - dataframe
        image_name_index - index of the column containing the image id (capture event id)
        url_col_index - index of the columns containing the url for the image id
        outpath - path on the local directory where the image has to be saved
    return: 
        Downloaded images in the path - outpath
    Usage:
        get_images_from_url(df3, image_name_index=0, url_col_index=6, outpath = '/Users/manishrai/Desktop/test_dir/')"""
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)): 
        ssl._create_default_https_context = ssl._create_unverified_context
        
        check = []
        
        for i in range(dataset.shape[0]):
            if dataset.iloc[i][image_name_index] not in check:
                j = 0
            if dataset.iloc[i][image_name_index] in check:
                j += 1 
            
            print('Processing image: %d' % i)
            
            urllib.request.urlretrieve(dataset.iloc[i][url_col_index], outpath+'{0}.jpg'.format(dataset.iloc[i][image_name_index] ))
            
def run_inference_graph_images(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS, \
                               NUM_CLASSES=1, TEST_IMAGE_PATHS, min_threshold, \
                               bb_outpath, PATH_TO_BB_HASHMAP):
    """This function takes in a list of image local-paths and runs it through the trained graph.
    Further, using the visualization function it draws bounding boxes on each of the images and 
    saves it in a local path.
    Arguments:
        PATH_TO_FROZEN_GRAPH - local path of the trained frozen graph, '/frozen_inference_graph.pb'
        PATH_TO_LABELS - local path of the labels (a mapping from class number to class name), 'label_map_focus.pbtxt'
        NUM_CLASSES - number of detection classes
        TEST_IMAGE_PATHS - list of test image local paths
        min_threshold - minimum score threshold for the bounding box to be considered
        bb_outpath - local path where to save the images with ounding poxes, /home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/snapshot-safari/snapshot-serengeti/subject_set_upload/
        PATH_TO_BB_HASHMAP - path where bounding box information for the subjects is saved
        
        """
    # Loading the frozen graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    bb_hashmap = {}
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        if (len(np.array(image).shape) == 3):
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
        
        # Considering the default dpi of matplotlib, calculating the figure size to save
            y0, x0, c = image_np.shape
            h = y0/72 # the default for dpi for matplotlib is 72
            w = x0/72 # the default for dpi for matplotlib is 72
            IMAGE_SIZE = (w, h)
        
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=8,
              min_score_thresh=min_threshold,
              skip_labels=True,
              skip_scores=True,
              agnostic_mode=True
            )

            # make a figure without the frame
            fig = plt.figure(frameon=False, figsize=IMAGE_SIZE)
            # make the content fill the whole figure
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            # draw your image
            ax.imshow(image_np)
            plt.savefig(os.path.join(bb_outpath, '{0}'.format(image_path[-14:]))) # saving image with boxes on the disk
            plt.gcf().clear()
            bb_hashmap[image_path[-14:]] = {
              'detection_boxes' : output_dict['detection_boxes'] [0:sum(output_dict['detection_scores']>=min_threshold)],
          'detection_scores' : output_dict['detection_scores'][0:sum(output_dict['detection_scores']>=min_threshold)]
        }
            
    with open(PATH_TO_BB_HASHMAP, 'w') as f:
        for key in bb_hashmap.keys():
            f.write("%s,%s\n"%(key,bb_hashmap[key]))
            
    return bb_hashmap