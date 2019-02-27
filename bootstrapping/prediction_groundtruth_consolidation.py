""" Compares the predictions and the ground truth classifications for 
bootstrapping.
"""
import pandas as pd
import json

def flag_df(df):
    if (11 <= df['prediction_counts'] <= 50):
        return '11-50'
    elif df['prediction_counts']>=51:
        return '51+'
    else:
        return df['prediction_counts']

def prediction_count_aggregation(dataframe):
    """ Takes in the predictions from the output of
    https://github.com/Manish-rai21bit/camera-trap-detection/blob/master/predictorExtractor_main.py
    
    and aggregate the data by filename and species
    """
    prediction_df = pd.read_csv(dataframe)
    
    # Aggregate predictions per image and species
    prediction_df_agg = pd.DataFrame(prediction_df.groupby(by=['filename', 'labels'], as_index=False)['score'].count())\
                    .rename(columns = {'score':'prediction_counts'}, index=str)
    # Rolling up the counts 11-50 and 51>
    prediction_df_agg['prediction_counts'] = prediction_df_agg.apply(flag_df, axis = 1)
    
    return prediction_df_agg


def process_grondtruth_classification_data(dataframe, 
                                           label_map_df):
    """Process the data """
    groundtruth_df = pd.read_csv(dataframe)
    # data preprocessing 
    groundtruth_df_img = groundtruth_df[['species', 'count', 'image1']]
    groundtruth_df_img = groundtruth_df_img.rename(columns = {'image1':'filename'}, index=str)

    groundtruth_df_img2 = groundtruth_df[['species', 'count', 'image2']]
    groundtruth_df_img2 = groundtruth_df_img2.rename(columns = {'image2':'filename'}, index=str)

    groundtruth_df_img3 = groundtruth_df[['species', 'count', 'image3']]
    groundtruth_df_img3 = groundtruth_df_img3.rename(columns = {'image3':'filename'}, index=str)

    groundtruth_df_img.append(groundtruth_df_img2)
    groundtruth_df_img.append(groundtruth_df_img3)

    # The groundtruth/classification data has a lot empty images. removing the empty images
    groundtruth_df_img = groundtruth_df_img[groundtruth_df_img['species']!= 'empty']

    # Bring in the label map information
    groundtruth_df_img = pd.merge(left=groundtruth_df_img,
                            right=label_map_df,
                            left_on=groundtruth_df_img['species'].str.lower(),
                            right_on=label_map_df['species'].str.lower(),
                            how='inner')
    groundtruth_df_img = groundtruth_df_img[['filename', 'species_x', 'labels', 'count']]\
                            .rename(columns = {'species_x':'species', 'count': 'groundtruth_counts'}, index=str)
    groundtruth_df_img['filename'] = [big_filename[4:-4]for big_filename in groundtruth_df_img['filename']]
    
    return groundtruth_df_img

def prediction_groundtruth_intersection_dataframe(groundtruth_dataframe, 
                                                  prediction_dataframe
                                                 ):
    """Returns dataframe with images in the intersection of groundtruth and prediction images
    prediction data aggregated by image, species and count, and groundtruth data
    list of images that intersect
    """
    intersection_images = list(set(groundtruth_dataframe.filename).intersection(set(prediction_dataframe.filename)))
    groundtruth_dataframe_intersection = groundtruth_dataframe.loc[groundtruth_dataframe['filename'].isin(intersection_images)]
    prediction_dataframe_intersection = prediction_dataframe.loc[prediction_dataframe['filename'].isin(intersection_images)]
    
    return groundtruth_dataframe_intersection, prediction_dataframe_intersection, intersection_images


def merged_groundtruth_prediction_dataframe(groundtruth_dataframe,
                                            prediction_dataframe,
                                            label_map_df,
                                            join_type='outer'):
    """returns dataframe with groundtruth dataframe and the predicted datframe merged"""
    groundtruth_prediction_dataframe = pd.merge(left=groundtruth_dataframe,
                                                right=prediction_dataframe,
                                                left_on=['filename', 'labels'],
                                                right_on=['filename', 'labels'],
                                                how=join_type)
    
    groundtruth_prediction_dataframe = groundtruth_prediction_dataframe[['filename', 'labels', 'groundtruth_counts','prediction_counts']]
    groundtruth_prediction_dataframe = pd.merge(left=groundtruth_prediction_dataframe,
                                                right=label_map_df,
                                                left_on=['labels'],
                                                right_on=['labels'],
                                                how='left')
    groundtruth_prediction_dataframe = groundtruth_prediction_dataframe[['filename', 'species', 'labels', 'groundtruth_counts','prediction_counts']]
    
    return groundtruth_prediction_dataframe