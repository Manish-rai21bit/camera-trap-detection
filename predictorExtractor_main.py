"""Main function for extracting TFRecord to a csv. 
Uses the helper modules predictor_extractor.py

python predictorExtractor_main.py \
    --tfrecord_path_list 'Predictions/snapshot_serengeti_s01_s06-0-10000.record' \
    --output_csv 'Predictions/snapshot_serengeti_test2.csv'
"""

import argparse

from data_prep.predictor_extractor import predictorExtractor


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



    kwargs = vars(parser.parse_args())

    predictorExtractor(**kwargs)