
import pandas as pd

# Consolidate all the prediction_groundtruth shards
def combine_pred_groundtruth_consolidated_files(filepath_list):
    """This function combines all the small shards pred_groundtruth_consolidate_snapshot_serengeti_s01_s06
    into a big file.
    Parameter: filepath for the pred_groundtruth_consolidate_snapshot_serengeti_s01_s06*
    Return: consolidated datframe
    """
    df_pred_gt_consolidated = pd.DataFrame()
    list_temp = []
    for i, filepath in enumerate(filepath_list):
        df_pred_gt_temp = pd.read_csv(filepath)
#         df_pred_gt_temp = df_pred_gt_temp.fillna(0)
        list_temp.append(df_pred_gt_temp.shape[0]) # for use to check the correct append. len(list_temp)== 400
        df_pred_gt_consolidated = df_pred_gt_consolidated.append(df_pred_gt_temp)
    df_pred_gt_consolidated = df_pred_gt_consolidated.reset_index()
    df_pred_gt_consolidated = df_pred_gt_consolidated[['filename', 'species', 'labels', 'groundtruth_counts', 'prediction_counts']]
    return df_pred_gt_consolidated

# Consolidate all the TF Record decoded shards
def combine_tfr_decoded_predictions(filepath_list):
    """This function combines all the small shards pred_groundtruth_consolidate_snapshot_serengeti_s01_s06
    into a big file.
    Parameter: filepath for the pred_groundtruth_consolidate_snapshot_serengeti_s01_s06*
    Return: consolidated datframe
    """
    df_pred_gt_consolidated = pd.DataFrame()
    list_temp = []
    for i, filepath in enumerate(filepath_list):
        df_pred_gt_temp = pd.read_csv(filepath)
#         df_pred_gt_temp = df_pred_gt_temp.fillna(0)
        list_temp.append(df_pred_gt_temp.shape[0]) # for use to check the correct append. len(list_temp)== 400
        df_pred_gt_consolidated = df_pred_gt_consolidated.append(df_pred_gt_temp)
    df_pred_gt_consolidated = df_pred_gt_consolidated.reset_index()
    df_pred_gt_consolidated = df_pred_gt_consolidated[['filename', 'labels', 'score','xmax','xmin','ymax','ymin']]
    return df_pred_gt_consolidated