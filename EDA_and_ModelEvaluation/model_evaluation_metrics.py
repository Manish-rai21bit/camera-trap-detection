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
            Precision
    """
    df_pred_gt_consolidated_inter = pred_groundtruth_consolidate_df.to_dict(orient='index')
    df_pred_gt_consolidated_dict = {}
    for k, v in df_pred_gt_consolidated_inter.items():
        if v['filename'] not in df_pred_gt_consolidated_dict.keys():
            df_pred_gt_consolidated_dict[v['filename']] = {}
            if pd.isnull(v['prediction_counts']) or pd.isnull(v['groundtruth_counts']):
                df_pred_gt_consolidated_dict[v['filename']]['correct_class'] = 0
                df_pred_gt_consolidated_dict[v['filename']]['incorrect_class'] = 1
            else:
                df_pred_gt_consolidated_dict[v['filename']]['correct_class'] = 1
                df_pred_gt_consolidated_dict[v['filename']]['incorrect_class'] = 0

        else:
            if pd.isnull(v['prediction_counts']) or pd.isnull(v['groundtruth_counts']):
                df_pred_gt_consolidated_dict[v['filename']]['correct_class'] += 0
                df_pred_gt_consolidated_dict[v['filename']]['incorrect_class'] += 1
            else:
                df_pred_gt_consolidated_dict[v['filename']]['correct_class'] += 1
                df_pred_gt_consolidated_dict[v['filename']]['incorrect_class'] += 0
    
    df_pred_gt_consolidated_leve1_class = pd.DataFrame(df_pred_gt_consolidated_dict).transpose().reset_index()
    # The overall classification accuracy
    correct_df = df_pred_gt_consolidated_leve1_class[(df_pred_gt_consolidated_leve1_class.correct_class>=1) \
                                        & (df_pred_gt_consolidated_leve1_class.incorrect_class==0)]
#     accuracy = sum(correct_df.correct_class)/len(set(df_pred_gt_consolidated_leve1_class.index))
    accuracy = correct_df.shape[0]/len(set(df_pred_gt_consolidated_leve1_class.index))
    precision = sum(df_pred_gt_consolidated_leve1_class.correct_class)/ \
                            (sum(df_pred_gt_consolidated_leve1_class.correct_class) + \
                             sum(df_pred_gt_consolidated_leve1_class.incorrect_class))
    print("The overall accuracy in classification: {0}".format(round(accuracy, 3)))
    print("The overall precision in classification: {0}".format(round(precision, 3)))
    return accuracy, precision

def get_binary_classifcation_species_level_perfomance(pred_groundtruth_consolidate_df):
    """
    1. b - Species level classification metric. 
    """
    # Overall Species Level
    y_true = [not(pd.isnull(val)) for val in pred_groundtruth_consolidate_df["groundtruth_counts"]]
    y_pred = [not(pd.isnull(val)) for val in pred_groundtruth_consolidate_df["prediction_counts"]]
    tn, fp, fn, tp = metric.confusion_matrix(y_true, y_pred).ravel()
    # For a classification task the recall is:
    recall = round(tp/(tp + fn), 3) # Correct
    precision = round(tp/(tp + fp), 3)
    accuracy = round((tp)/(tp + fn), 3)
    f1_score = round(2*recall*precision/(recall + precision), 3)
    
    print("Level 1: Species Level Overall")
    print("Recall: {0}".format(recall))
    print("Precision: {0}".format(precision))
    print("F1-Score: {0}".format(f1_score))
    print("Accuracy: {0}".format(accuracy))
    
    # Per Species Level
    species_level_performance_binary = {}
    for species in set(pred_groundtruth_consolidate_df['species']):
        species_level_performance_binary[species] = {}
        error = False
        df_temp = pred_groundtruth_consolidate_df[pred_groundtruth_consolidate_df['species']==species]
        y_true = [not(pd.isnull(val)) for val in df_temp["groundtruth_counts"]]
        y_pred = [not(pd.isnull(val)) for val in df_temp["prediction_counts"]]
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