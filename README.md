# Deep Learning for species detection in camera-trap data
Consists of 2 modules:
1. Modelling <br>
2. Zooniverse Research/Analysis <br>

Camera-traps are a rich data source for ecologists and is used to get a sense of the species population in the wild.

Steps for prediction are:
1. Create the tensorflow record for training and testing data. Here is the code used to generate the [tf_record](https://github.com/Manish-rai21bit/camera-trap-detection/blob/master/Data_Model.ipynb). <br>
Once the data is created for training and testing we can use this data to train our model and test the performance.<br>
2. A pre-trained Faster R-CNN is used as the base and retrained for our specific task of animal detection. The pre-trained model can be downloaded from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). <br>
3. The directory structure for the training and prediction can be made similar to the [training_demo](https://github.com/Manish-rai21bit/camera-trap-detection/tree/master/training_demo). This directory can be placed inside the tensorflow/model directory from where we can trigger the below model training pipeline and graph export pipeline. <br>

Model training pipeline to run as a sparete module<br>

cd /Users/manishrai/Desktop/UMN/Research/Zooniverse/Code/tensorflow/models/my_workspace/training_demo <br>

PIPELINE_CONFIG_PATH='training/faster_rcnn_resnet101_coco.config'<br>

MODEL_DIR='/Users/manishrai/Desktop/UMN/Research/Zooniverse/Code/tensorflow/models/my_workspace/training_demo/training/'<br>

NUM_TRAIN_STEPS=900000 <br>

NUM_EVAL_STEPS=2000 <br>

python model_main.py \ <br>
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \ <br>
    --model_dir=${MODEL_DIR} \ <br>
    --num_train_steps=${NUM_TRAIN_STEPS} \ <br>
    --num_eval_steps=${NUM_EVAL_STEPS} \ <br>
    --alsologtostderr <br>


python export_inference_graph.py \ <br>
    --input_type image_tensor \ <br>
    --pipeline_config_path training/faster_rcnn_resnet101_coco.config \ <br>
    --trained_checkpoint_prefix training/model.ckpt-500 \ <br>
    --output_directory trained-inference-graphs/output_inference_graph_v1.pb <br>



### Pipeline for boorstrapping data preparation for feeding back to training loop
**Predicted TFRecord to CSV decoder:** <br>
python predictorExtractor_main.py \
    --tfrecord_path_list '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_s01_s06-0-10000.record' \
    --output_csv '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_s01_s06-0-10000.csv'

**Consolidated view with ground truth and predicted species count per image:**<br>
python prediction_groundtruth_consolidation_main.py \
    --prediction_csv_path '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_s01_s06-0-10000.csv' \
    --groundtruth_csv_path '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/db_export_season_all_cleaned.csv' \
    --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
    --outfile '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/pred_groundtruth_consolidate_csv.csv'

**Create CSV with bounding boxes for TFRecord encoding**<br>
python bootstrapping_data_prep_main.py \
    --pred_groundtruth_consolidate_csv '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/pred_groundtruth_consolidate_csv.csv' \
    --prediction_csv_path '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_s01_s06-0-10000.csv' \
    --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
    --outfile '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/bootstrap_data_snapshot_serengeti_s01_s06-0-10000.csv'

**move some files around and run this on MSI** <br>
**TFRecord encoder for bootstrapped bounding box**<br>
python tfr_encoder_for_bootstarpping_main.py \
    --image_filepath '/panfs/roc/groups/5/packerc/shared/albums/SER/' \
    --bounding_box_csv 'bootstrap_data_snapshot_serengeti_s01_s06-0-10000.csv' \
    --label_map_json '/home/packerc/rai00007/camera-trap-detection/data/LILA/label_map.json' \
    --output_tfrecord_file 'out_tfr_file.record'
