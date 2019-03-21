"""Main function for extracting TFRecord to a csv. 
Uses the helper modules predictor_extractor.py"""

import argparse
import pandas as pd

from data_prep.predictor_extractor import predictorExtractor
import data_prep.data_prep_utils as dataprep_utils
import bootstrapping.prediction_groundtruth_consolidation as pgc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tfrecord_path_list", nargs='+', type=str, required=True,
        help="Path to TFRecord files")
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="output csv file"
        )
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="batch size")
    parser.add_argument(
        "--score_threshold", type=float, default=0.5,
        help="score thresholds to write to csv")
    parser.add_argument(
        "--discard_image_pixels", type=bool, default=True,
        help="True to discard the pixel encodings or when pixel encodings are not present in the datafile")
    parser.add_argument(
        "--groundtruth_csv_path", type=str, required=True,
        help="path to the groundtruth file"
    )
    parser.add_argument(
        "--label_map_json", type=str, required=True,
        help="path to the label map json file")
    parser.add_argument(
        "--is_training", type=bool, default=True,
        help="if the data is for the training purposes of bootstrapping step")





#     kwargs = vars(parser.parse_args())
    args = parser.parse_args()
    
    label_map_df = pd.DataFrame.from_dict(dataprep_utils.get_label_map_from_json(args.label_map_json), orient='index').reset_index()
    label_map_df.columns=['species', 'labels']

    groundtruth_df_img = pgc.process_grondtruth_classification_data(args.groundtruth_csv_path, label_map_df)
    groundtruth_dict = groundtruth_df_img.to_dict(orient='index')
    groundtruth_consolidated_dict = {}
    
    for k, v in groundtruth_dict.items():
        if v['filename'] not in groundtruth_consolidated_dict.keys():
            groundtruth_consolidated_dict[v['filename']] = v['groundtruth_counts']
        elif v['groundtruth_counts'] == '11-50' and groundtruth_consolidated_dict[v['filename']] != '51+':
            groundtruth_consolidated_dict[v['filename']] = '11-50'
        elif v['groundtruth_counts'] != '51+' and groundtruth_consolidated_dict[v['filename']] == '11-50':
            groundtruth_consolidated_dict[v['filename']] = '11-50'
        elif v['groundtruth_counts'] == '51+' or groundtruth_consolidated_dict[v['filename']] == '51+':
            groundtruth_consolidated_dict[v['filename']] = '51+'
        else:
            groundtruth_consolidated_dict[v['filename']] = int(groundtruth_consolidated_dict[v['filename']])+int(v['groundtruth_counts'])
            
    predictorExtractor(args.tfrecord_path_list, args.output_csv, groundtruth_consolidated_dict)
#     predictorExtractor(**kwargs)