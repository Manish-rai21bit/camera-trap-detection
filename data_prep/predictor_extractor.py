"""Predictor Extractor reads the TFRecords with predictions and outputs the csv file with the predictions """
import tensorflow as tf
import pandas as pd

tf.enable_eager_execution()

def decode_record(serialized_example, discard_image_pixels=True):
    """This section contains the helper functions needed for 
    decoding TFRecord...
    1. decode_record: This function decodes a TFexample with the following features
                   'image/filename' - a fixed length feature with the file name
                   'image/encoded' - a fixed length feature with image encodings
                   'image/format' - a fixed length features with image format
                   'image/detection/bbox/xmin' - a varaible name feature with normalized xmin values 
                   'image/detection/bbox/xmax' - normalized xmax values
                   'image/detection/bbox/ymin' - normalized ymin values
                   'image/detection/bbox/ymax' - normalized ymax values
                   'image/detection/label' - bounding box labels
                   'image/detection/score' - prediction score

    """


    context_features = {
                        'image/filename': tf.FixedLenFeature([], tf.string),
                        # 'image/encoded': tf.FixedLenFeature([], tf.string),
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


def predictorExtractor(tfrecord_path_list,
                       output_csv,
                       groundtruth_consolidated_dict,
                       discard_image_pixels=True,
                       batch_size=512, 
                       score_threshold=0.5,
                       score_threshold_lower_bound_for_bootstrap=0.5,
                       is_training='True'):
    """
    This requires a dicrionary with groudtruth file name and total animals in it
    """
    
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_path_list)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.repeat(1)
    dataset = dataset.map(lambda x: decode_record(serialized_example=x)).batch(batch_size)

    xmins_d, ymins_d, xmaxs_d, ymaxs_d, labels_d, scores, filenames = [], [], [], [], [], [], []
    filenames_without_predictions = []


    for i, (context, sequence) in enumerate(dataset):
        batch_shape = context['image/detection/bbox/xmin'].dense_shape
        #context['image/detection/bbox/xmin'][2]
        # if discard_image_pixels!=True:
        # img = context['image/encoded']
        filename = context['image/filename']
        # Features added during the detection phase 
        xmin_d = tf.sparse_tensor_to_dense(context['image/detection/bbox/xmin'])
        ymin_d = tf.sparse_tensor_to_dense(context['image/detection/bbox/ymin'])
        xmax_d = tf.sparse_tensor_to_dense(context['image/detection/bbox/xmax'])
        ymax_d = tf.sparse_tensor_to_dense(context['image/detection/bbox/ymax'])
        label_d = tf.sparse_tensor_to_dense(context['image/detection/label'])
        score = tf.sparse_tensor_to_dense(context['image/detection/score'])


        for rec_i in range(0, int(batch_shape[0])):
            if is_training=='True':
                num_box = groundtruth_consolidated_dict[filename[rec_i].numpy().decode('utf-8')]
                if num_box in ['11-50', '51+']:
                    score_threshold = 0.25
                else:
                    # max(min(scores), lower_bound) to enforce quality of boxes
                    score_threshold = max(min(score[rec_i, 0:int(num_box)]), score_threshold_lower_bound_for_bootstrap)
            box_counter = 0
            for box_i in range(0, int(batch_shape[1])):
                if score[rec_i, box_i] < score_threshold:
                    continue
                xmins_d.append(xmin_d[rec_i, box_i].numpy())
                ymins_d.append(ymin_d[rec_i, box_i].numpy())
                xmaxs_d.append(xmax_d[rec_i, box_i].numpy())
                ymaxs_d.append(ymax_d[rec_i, box_i].numpy())
                labels_d.append(int(label_d[rec_i, box_i].numpy()))
                scores.append(score[rec_i, box_i].numpy())
                filenames.append(filename[rec_i].numpy().decode('utf-8'))

                box_counter += 1

            if box_counter == 0:
                filenames_without_predictions.append(filename[rec_i].numpy().decode('utf-8'))

        print('Batch: {0} finished'.format(i))
    
     # Create pandas dataframe
    df_predictions = pd.DataFrame({'labels':labels_d, 
                                   'filename':filenames,
                                   'score': scores,
                                   'xmin': xmins_d,
                                   'ymin': ymins_d,
                                   'xmax': xmaxs_d,
                                   'ymax': ymaxs_d})

    df_predictions = df_predictions.append(pd.DataFrame({'filename':filenames_without_predictions}))
    df_predictions = df_predictions.round({'score':2})
    # write predictions to csv
    df_predictions.to_csv(output_csv, index=False)