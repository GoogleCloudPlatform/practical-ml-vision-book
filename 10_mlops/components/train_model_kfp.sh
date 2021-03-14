#!/bin/bash -x

INPUT_TOPDIR=$1
REGION=$2
JOB_DIR=$3
COMPONENT_OUT=$4

cd /src/practical-ml-vision-book/07_training/serverlessml

PACKAGE_PATH="./flowers"
MODULE_NAME="flowers.classifier.train"

gsutil ls $INPUT_TOPDIR/train-00000-*

gcloud ai-platform local train \
        --package-path $PACKAGE_PATH \
        --module-name $MODULE_NAME \
        --job-dir ${JOB_DIR} \
        -- \
        --input_topdir $INPUT_TOPDIR \
        --pattern='-00000-*' \
        --num_epochs=3 --l2 0 --with_color_distort False --crop_ratio 0.8

mkdir -p $(dirname $COMPONENT_OUT)
echo "${JOB_DIR}/flowers_model" > $COMPONENT_OUT
