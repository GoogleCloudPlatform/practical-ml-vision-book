#!/bin/bash -x

INPUT_TOPDIR=$1
REGION=$2
JOB_DIR=$3
COMPONENT_OUT=$4

cd /src/practical-ml-vision-book/07_training/serverlessml

PACKAGE_PATH="./flowers"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="flowers_$now"
MODULE_NAME="flowers.classifier.train"

# gpus on one machine
DISTR="gpus_one_machine"
CONFIG="${DISTR}.yaml"

gcloud ai-platform jobs submit training ${JOB_NAME}_${DISTR} \
        --config ${CONFIG} --region ${REGION} \
        --package-path $PACKAGE_PATH \
        --module-name $MODULE_NAME \
        --job-dir ${JOB_DIR} \
        --runtime-version 2.3 --python-version 3.7 \
        --stream-logs \
        -- \
        --input_topdir $INPUT_TOPDIR \
        --pattern='-*' \
        --num_epochs=20 --distribute $DISTR --l2 0 --with_color_distort False

mkdir -p $(dirname $COMPONENT_OUT)
echo "${JOB_DIR}/flowers_model" > $COMPONENT_OUT
