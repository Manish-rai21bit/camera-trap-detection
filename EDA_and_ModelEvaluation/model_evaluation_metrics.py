# importing the necessary librabies
import csv, glob
import pandas as pd
import numpy as np
import sklearn.metrics as metric
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


"""Evaluation at 3 levels:
1. Binary Classification
    a. Overall level - correctly find all the animals in the image
    b. Species level - Correctly find animal in the image
2. Count Level (wip)

3. With IoU (wip)
"""

def get_binary_classifcation_overall_perfomance(pred_groundtruth_consolidate_df):
    """1. a:
    Calculates the overall accuracy of the model. 
    input: DataFrame(output of this code - prediction_groundtruth_consolidation_main) 
            with predictions and groundtruth consolidated.
    output: Accuracy on classification for images. 
            Ex - if the GroundTruth has ['lion', 'cats'] and predictions has ['lion', 'cats'] - Correct,
            else Incorrect
    """
    df_pred_gt_consolidated_inter = pred_groundtruth_consolidate_df.to_dict(orient='index')
    df_pred_gt_consolidated_dict = {}
    for k, v in df_pred_gt_consolidated_inter.items():

        if v['filename'] not in df_pred_gt_consolidated_dict.keys():
            df_pred_gt_consolidated_dict[v['filename']] = {}
            if (v['prediction_counts'] not in ['11-50', '51+']) and (v['groundtruth_counts'] not in ['11-50', '51+']):
                if (int(v['prediction_counts'])>0) and (int(v['groundtruth_counts'])>0):
                    df_pred_gt_consolidated_dict[v['filename']]['correct_class'] = 1
                    df_pred_gt_consolidated_dict[v['filename']]['incorrect_class'] = 0
                else:
                    df_pred_gt_consolidated_dict[v['filename']]['correct_class'] = 0
                    df_pred_gt_consolidated_dict[v['filename']]['incorrect_class'] = 1
            else:
                if (v['prediction_counts'] == v['groundtruth_counts']):
                    df_pred_gt_consolidated_dict[v['filename']]['correct_class'] = 1
                    df_pred_gt_consolidated_dict[v['filename']]['incorrect_class'] = 0
                else:
                    df_pred_gt_consolidated_dict[v['filename']]['correct_class'] = 0
                    df_pred_gt_consolidated_dict[v['filename']]['incorrect_class'] = 1

        else:
            if v['prediction_counts']==v['groundtruth_counts']:
                df_pred_gt_consolidated_dict[v['filename']]['correct_class'] += 1
                df_pred_gt_consolidated_dict[v['filename']]['incorrect_class'] += 0
            else:
                df_pred_gt_consolidated_dict[v['filename']]['correct_class'] += 0
                df_pred_gt_consolidated_dict[v['filename']]['incorrect_class'] += 1

    # The overall classification accuracy
    correct_df = df_pred_gt_consolidated_leve1_class[(df_pred_gt_consolidated_leve1_class.correct_class>=1) \
                                        & (df_pred_gt_consolidated_leve1_class.incorrect_class==0)]
    accuracy = correct_df.shape[0]/len(set(pred_groundtruth_consolidate_df.filename))
    print("The overall accuracy in classification: {0}".format(round(accuracy, 3)))
    
    return accuracy

def get_binary_classifcation_species_level_perfomance(pred_groundtruth_consolidate_df):
    """
    1. b - Species level classification metric. 
    """
    species_level_performance_binary = {}
    for species in set(df_pred_gt_consolidated['species']):
        species_level_performance_binary[species] = {}
        error = False
        df_temp = pred_groundtruth_consolidate_df[pred_groundtruth_consolidate_df['species']==species]
        y_true = [val != 0 for val in df_temp["groundtruth_counts"]]
        y_pred = [val != 0 for val in df_temp["prediction_counts"]]
        try:
            tn, fp, fn, tp = metric.confusion_matrix(y_true, y_pred).ravel()
        except Exception:
    #         y_true = [2] # to know why this exception run this code
    #         y_pred = [2]
    #         int(confusion_matrix(y_true, y_pred))
            tn, fp, fn, tp = 0, 0, 0, int(metric.confusion_matrix(y_true, y_pred))
            pass

        species_level_performance_binary[species]['TP'] = tp
        species_level_performance_binary[species]['FP'] = fp
        species_level_performance_binary[species]['TN'] = tn
        species_level_performance_binary[species]['FN'] = fn
        
    return species_level_performance_binary