#!/usr/bin/env bash
PYTHONPATH="/home/jupyter/tensorflow-model-garden"
MODEL_DIR="gs://ml1-demo-martin/arthropod_jobs"
TRAIN_FILE_PATTERN="gs://practical-ml-vision-book/arthropod_detection_tfr/size_1024x724/*.tfrec"
#EVAL_FILE_PATTERN="<path to the TFRecord validation data>"
#VAL_JSON_FILE="<path to the validation annotation JSON file>"
python3 ~/tensorflow-model-garden/official/vision/detection/main.py \
  --strategy_type=tpu \
  --tpu="${TPU_NAME?}" \
  --model_dir="${MODEL_DIR?}" \
  --mode=train \
  --training_file_pattern=${TRAIN_FILE_PATTERN?} \
  --params_override="{ type: retinanet, architecture: {backbone: spinenet, multilevel_features: identity, use_bfloat16: False}, spinenet: {model_id: '49'}}"