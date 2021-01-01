#!/bin/bash

MODEL_NAME=flowers
VERSION_NAME=txf
BEST_TRIAL='33'

REGION='us-central1'  # make sure you have GPU/TPU quota in this region
BUCKET='ai-analytics-solutions-kfpdemo'
DISTR='gpus_one_machine'
MODEL_LOCATION="gs://${BUCKET}/flowers_${DISTR}/${BEST_TRIAL}/flowers_model"

if [[ $(gcloud ai-platform models list --format='value(name)' | grep $MODEL_NAME) ]]; then
    echo "The model named $MODEL_NAME already exists."
else
    # create model
    echo "Creating $MODEL_NAME model now."
    gcloud ai-platform models create --regions=$REGION $MODEL_NAME
fi

if [[ $(gcloud ai-platform versions list --model $MODEL_NAME --format='value(name)' | grep $VERSION_NAME) ]]; then
    echo "Deleting already the existing model $MODEL_NAME:$VERSION_NAME ... "
    gcloud ai-platform versions delete --model=$MODEL_NAME $VERSION_NAME
    echo "Please run this script again if you don't see a Creating message ... "
    sleep 2
fi

# create model
echo "Creating $MODEL_NAME:$VERSION_NAME"
gcloud ai-platform versions create --model=$MODEL_NAME $VERSION_NAME --async \
       --framework=tensorflow --python-version=3.7 --runtime-version=2.3 \
       --origin=$MODEL_LOCATION --staging-bucket=gs://$BUCKET
