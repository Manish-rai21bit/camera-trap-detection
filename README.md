# camera-trap-detection

Model training pipeline

cd /Users/manishrai/Desktop/UMN/Research/Zooniverse/Code/tensorflow/models/my_workspace/training_demo

PIPELINE_CONFIG_PATH='training/faster_rcnn_resnet101_coco.config'

MODEL_DIR='/Users/manishrai/Desktop/UMN/Research/Zooniverse/Code/tensorflow/models/my_workspace/training_demo/training/'

NUM_TRAIN_STEPS=500

NUM_EVAL_STEPS=20

python model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH}  \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr


python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/faster_rcnn_resnet101_coco.config \
    --trained_checkpoint_prefix training/model.ckpt-500 \
    --output_directory trained-inference-graphs/output_inference_graph_v1.pb
