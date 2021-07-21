#!/bin/bash

ENDPOINT_NAME="flowers_endpoint"
MODEL_NAME="flowers_txf"
BEST_TRIAL='33'

REGION='us-central1'  # make sure you have GPU/TPU quota in this region
BUCKET='ai-analytics-solutions-kfpdemo'
DISTR='gpus_one_machine'
MODEL_LOCATION="gs://${BUCKET}/flowers_${DISTR}/${BEST_TRIAL}/flowers_model"
IMAGE_URI="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-1:latest"

echo "Deploying model $MODEL_NAME"

if [[ $(gcloud ai endpoints list --region=$REGION --format="value(display_name)" | grep $ENDPOINT_NAME) ]]; then
    echo "The endpoint named $ENDPOINT_NAME already exists."
else
    # Create endpoint.
    echo "Creating $ENDPOINT_NAME endpoint now."
    gcloud ai endpoints create \
      --region=$REGION \
      --display-name=$ENDPOINT_NAME
fi

ENDPOINT_ID=$(gcloud ai endpoints list --region=$REGION --format="value(name)" --filter="displayName=$ENDPOINT_NAME")
echo "The endpoint_id is $ENDPOINT_ID"

if [[ $(gcloud ai models list --region=$REGION --format="value(display_name)" | grep $MODEL_NAME) ]]; then
    echo "The model named $MODEL_NAME already exists."
else
    # Upload model.
    echo "Uploading $MODEL_NAME model now."
    gcloud ai models upload \
      --region=$REGION \
      --display-name=$MODEL_NAME \
      --container-image-uri=$IMAGE_URI \
      --artifact-uri=$MODEL_LOCATION
fi

MODEL_ID=$(gcloud ai models list --region=$REGION --format="value(name)" --filter="displayName=$MODEL_NAME")
echo "The model_id is $MODEL_ID"

echo "Deploying model now"
gcloud ai endpoints deploy-model $ENDPOINT_ID\
  --region=$REGION \
  --model=$MODEL_ID \
  --display-name=$MODEL_NAME \
  --traffic-split=0=100

