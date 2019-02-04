"""Main function to create a csv with bounding box annotations 
that will be used to create a TFRecord file for model training

python bootstrapping/bootstrapping_data_prep_main.py \
--pred_groundtruth_consolidate_csv '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/test.csv' \
--prediction_csv_path '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_test2.csv' \
--label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
--outfile '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_test3.csv'
"""

import argparse 
import pandas as pd

import bootstrapping.bootstrapping_data_prep as bdp
import data_prep.data_prep_utils as dataprep_utils

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--pred_groundtruth_consolidate_csv", type=str, required=True,
    help="path to the csv with predicted and groundtruth box counts info. Fields are\
        filename,species,labels,groundtruth_counts,prediction_counts"
    )
    parser.add_argument(
    "--prediction_csv_path", type=str, required=True,
    help="path to the csv with the predicted bounding box. This is the output of the code \
    https://github.com/Manish-rai21bit/camera-trap-detection/blob/master/prediction_groundtruth_consolidation_main.py"
    )
    parser.add_argument(
    "--label_map_json", type=str, required=True,
    help="path to the label map json"
    )
    parser.add_argument(
    "--outfile", type=str, required=True,
    help="path to the csv where the save the csv ready for TFRecord creation")
    
    args = parser.parse_args()

    pred_groundtruth_consolidate_dict = bdp.pred_groundtruth_consolidate_csv_to_dict(args.pred_groundtruth_consolidate_csv)
    correct_list, corrected_image_species_list, incorrect_list = bdp.get_correct_incorrect_images(pred_groundtruth_consolidate_dict)

    label_map = dataprep_utils.get_label_map_from_json(args.label_map_json)

    correct_predicted_final_df = bdp.training_data_prep_from_correct_prediction(args.prediction_csv_path, 
                                                    correct_list, 
                                                    corrected_image_species_list, \
                                                    label_map 
                                                  )
    print("Writing File to location: ", args.outfile)
    correct_predicted_final_df.to_csv(args.outfile, index=False)