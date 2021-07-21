#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="flowers_$now"
REGION="us-central1"  # make sure you have GPU/TPU quota in this region
PROJECT=""  # TODO: Add your project here!
BUCKET="ai-analytics-solutions-kfpdemo"
JOB_DIR=gs://${BUCKET}/flowers
PYTHON_PACKAGE_URIS=gs://${BUCKET}/flowers-1.0.tar.gz

# cpu only
DISTR="cpu"

# gpus on one machine
# DISTR="gpus_one_machine"

# multiworker needs virtual epochs
#DISTR="gpus_multiple_machines"
#EXTRAS="--num_training_examples 4000"

#DISTR="tpu_caip"
#REGION="us-central1"

CONFIG=${DISTR}.yaml

# hyperparameter tuning
#CONFIG=hparam.yaml
#DISTR="gpus_one_machine"

# hyperparameter tuning
# CONFIG=hparam-continue.yaml  # note job identifier here
# DISTR="gpus_one_machine"
# EXTRAS="--l2 0 --with_color_distort False"

gcloud ai custom-jobs create \
--region=${REGION} \
--project=${PROJECT} \
--python-package-uris=${PYTHON_PACKAGE_URIS} \
--config=${CONFIG} \
--display-name=${JOB_NAME}_${DISTR}