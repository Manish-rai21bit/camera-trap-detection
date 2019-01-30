""" Main function for generating consolidated view for the
groundtruth and predictions.

python prediction_groundtruth_consolidation_main.py \
--prediction_csv_path '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_test2.csv' \
--groundtruth_csv_path '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/db_export_season_all_cleaned.csv' \
--label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
--outfile '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/test.csv'
"""
import argparse 
import pandas as pd

import bootstrapping.prediction_groundtruth_consolidation as pgc
import data_prep.data_prep_utils as dataprep_utils


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--prediction_csv_path", type=str, required=True,
    help="path to the csv with predicted bounding oxes per record"
    )
    parser.add_argument(
    "--groundtruth_csv_path", type=str, required=True,
    help="path to the groundtruth classification data from MSI"
    )
    parser.add_argument(
    "--label_map_json", type=str, required=True,
    help="path to json with label maps")
    parser.add_argument(
    "--outfile", type=str, required=True,
    help="path of the output csv to be created")
    
    args = parser.parse_args()
    
    label_map_df = pd.DataFrame.from_dict(dataprep_utils.get_label_map_from_json(args.label_map_json), orient='index').reset_index()
    label_map_df.columns=['species', 'labels']

    prediction_df_agg = pgc.prediction_count_aggregation(args.prediction_csv_path)
    groundtruth_df_img = pgc.process_grondtruth_classification_data(args.groundtruth_csv_path, label_map_df)

    gt_i, pred_i, intersection_images = pgc.prediction_groundtruth_intersection_dataframe(groundtruth_df_img, 
                                                      prediction_df_agg
                                                     )
    df = pgc.merged_groundtruth_prediction_dataframe(gt_i, \
                                            pred_i, \
                                            join_type='outer')
    print("Writing File to location: ", args.outfile)
    df.to_csv(args.outfile, index=False)