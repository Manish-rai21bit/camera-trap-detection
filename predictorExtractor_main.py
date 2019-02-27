"""Main function for extracting TFRecord to a csv. 
Uses the helper modules predictor_extractor.py"""

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
    parser.add_argument(
        "--discard_image_pixels", type=bool, default=True,
        help="True to discard the pixel encodings or when pixel encodings are not present in the datafile")



    kwargs = vars(parser.parse_args())

    predictorExtractor(**kwargs)