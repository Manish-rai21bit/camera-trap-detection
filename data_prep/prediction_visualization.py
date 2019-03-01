
"""Visualization API for:
1. Predictions
"""

import tensorflow as tf
import os, csv, io, sys
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import pyplot as plt
matplotlib.use('Agg')

tf.enable_eager_execution()

sys.path.append('/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/')

from data.utils import dataset_util


def decode_record(serialized_example):
    """Decode the TFRecord data for each example. """
    context_features = {
                        'image/filename': tf.FixedLenFeature([], tf.string),
                        'image/encoded': tf.FixedLenFeature([], tf.string),
                        'image/format': tf.FixedLenFeature([], tf.string),
                        "image/detection/bbox/xmin" : tf.VarLenFeature(tf.float32),
                        "image/detection/bbox/xmax" : tf.VarLenFeature(tf.float32),
                        "image/detection/bbox/ymin" : tf.VarLenFeature(tf.float32),
                        "image/detection/bbox/ymax" : tf.VarLenFeature(tf.float32),
                        "image/detection/label" : tf.VarLenFeature(tf.int64),
                        "image/detection/score" : tf.VarLenFeature(tf.float32)
                    }


    context, sequence = tf.parse_single_sequence_example(serialized=serialized_example,
                                              context_features=context_features,
#                                               sequence_features=sequence_features,
                                              example_name=None,
                                              name=None)

    return ({k: v for k, v in context.items()},{k: v for k, v in sequence.items()})


def plot_images_with_bbox(filename_list, outfile, inv_label_map, num_batches=1, 
                          score_threshold=0.5, batch_size= 2):
    """Plot n images with bounding boxes and save it in the location givenself.
    filename_list : list of TFRecords
    """

    # Create a tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices(filename_list)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.map(lambda x: decode_record(serialized_example=x)).batch(batch_size)

    for i, (context, sequence) in enumerate(dataset):
        if i<num_batches:
            batch_shape = context['image/detection/bbox/xmin'].dense_shape
            filename = context['image/filename']
            xmin_d = tf.sparse_tensor_to_dense(context['image/detection/bbox/xmin'])
            ymin_d = tf.sparse_tensor_to_dense(context['image/detection/bbox/ymin'])
            xmax_d = tf.sparse_tensor_to_dense(context['image/detection/bbox/xmax'])
            ymax_d = tf.sparse_tensor_to_dense(context['image/detection/bbox/ymax'])
            label_d = tf.sparse_tensor_to_dense(context['image/detection/label'])
            score = tf.sparse_tensor_to_dense(context['image/detection/score'])

            for rec_i in range(0, int(batch_shape[0])):
                xmins_d, ymins_d, xmaxs_d, ymaxs_d, labels_d, scores, filenames = [], [], [], [], [], [], []
                for box_i in range(0, int(batch_shape[1])):
                    if score[rec_i, box_i] < score_threshold:
                        continue

                    img = context['image/encoded'][rec_i]
                    encoded_jpg_io = io.BytesIO(img.numpy())
                    image = Image.open(encoded_jpg_io)
                    width, height = image.size

                    xmins_d.append((xmin_d[rec_i, box_i].numpy())*width)
                    ymins_d.append((ymin_d[rec_i, box_i].numpy())*height)
                    xmaxs_d.append((xmax_d[rec_i, box_i].numpy())*width)
                    ymaxs_d.append((ymax_d[rec_i, box_i].numpy())*height)
                    labels_d.append(int(label_d[rec_i, box_i].numpy()))
                    scores.append(score[rec_i, box_i].numpy())
                    filenames.append(filename[rec_i].numpy().decode('utf-8'))

                # Create figure and axes
                fig,ax = plt.subplots(1)
                fig.set_size_inches(10, 8)
                # Display the image
                ax.set_title(filename[rec_i].numpy().decode('utf-8'))
                ax.imshow(image)
                for s in range(len(xmins_d)):
                    rect = patches.Rectangle((xmins_d[s],ymins_d[s]),(xmaxs_d[s]-xmins_d[s]), \
                                             (ymaxs_d[s] - ymins_d[s]),linewidth=2,edgecolor='b',facecolor='none')
                    ax.add_patch(rect)

                    rx, ry = rect.get_xy()
                    cx = rx # + rect.get_width()/2.0
                    cy = ry # + rect.get_height()/2.0

                    ax.annotate((inv_label_map[labels_d[s]], str(scores[s])+str('%')), (cx, cy), color='b', weight='bold', 
                                fontsize=8, ha='left', va='top') 
                fig.savefig(os.path.join(outfile, filename[rec_i].numpy().decode('utf-8').split('/')[-1]))
                plt.clf