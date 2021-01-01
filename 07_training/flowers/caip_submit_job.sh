#!/bin/bash

PACKAGE_PATH="./flowers"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="flowers_$now"
MODULE_NAME="flowers.classifier.train"
JOB_DIR="gs://ai-analytics-solutions-kfpdemo/flowers"
REGION='us-west1'  # make sure you have GPU/TPU quota in this region

#gcloud ai-platform local train --package-path $PACKAGE_PATH --module-name $MODULE_NAME --job-dir $JOB_DIR 
#exit

# cpu only
#DISTR="cpu"
DISTR="gpus_one_machine"
#DISTR="gpus_multiple_machines"

#DISTR="tpu_caip"
#REGION="us-central1"

gcloud ai-platform jobs submit training ${JOB_NAME}_${DISTR} \
        --config ${DISTR}.yaml --region ${REGION} \
        --package-path $PACKAGE_PATH \
        --module-name $MODULE_NAME \
        --job-dir ${JOB_DIR}_${DISTR} \
        --runtime-version 2.3 --python-version 3.7 \
        -- \
        --pattern='-*' \
        --num_epochs=20 --distribute $DISTR
