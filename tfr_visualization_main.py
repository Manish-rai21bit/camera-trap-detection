"""Main function for visualizing the predictions and saving 
them to a directory

python tfr_visualization_main.py \
--filename_list '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_s01_s06-0-10000.record' \
--outfile '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/test_images/' \
--label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
--num_batches 256
"""

import json, argparse
import matplotlib
matplotlib.use('Agg')


import data_prep.tfr_visualization as visual

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename_list", nargs='+', type=str, required=True,
        help="Path to TFRecord files. In form of list")
    parser.add_argument(
        "--outfile", type=str, required=True,
        help="output directory of the image to be saved"
        )
    parser.add_argument(
        "--label_map_json", type=str, required=True,
        help="label map json"
    )
    parser.add_argument(
        "--num_batches", type=int, default=1,
        help="number of batches to save. batch size = 2"
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.5,
        help="threshold of the detected box to be plotted"
    )
    parser.add_argument(
        "--TFRecord_type", type=str, required=True,
        help="Type of the TF Record. \
             Pred: for TFRecord with Predictions, \
             Train: for TFRecord with Training dataset"
    )

    args = parser.parse_args()

    with open(args.label_map_json, 'r') as f:
         label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    
    if args.TFRecord_type=='Pred':
        visual.plot_images_with_bbox_pred(args.filename_list, 
                                          args.outfile, 
                                          inv_label_map, 
                                          args.num_batches, 
                                          score_threshold = args.score_threshold)
    elif args.TFRecord_type=='Train':
        visual.plot_images_training_with_bbox(args.filename_list, 
                                          args.outfile, 
                                          inv_label_map, 
                                          args.num_batches)
    elif args.TFRecord_type=='Raw':
        visual.plot_images_raw(args.filename_list, 
                               args.outfile,
                              args.num_batches)