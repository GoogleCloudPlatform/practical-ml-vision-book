#!/bin/bash

REGION="us-central1"  # make sure you have GPU/TPU quota in this region
ENDPOINT_NAME="flowers_endpoint"
MODEL_NAME="flowers"
MODEL_LOCATION="gs://practical-ml-vision-book-data/flowers_5_trained"
IMAGE_URI="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-1:latest"

for i in "$@"
do
case $i in
        -r=*|--region=*) REGION="${i#*=}"; shift ;;
        -e=*|--endpoint_name=*) ENDPOINT_NAME="${i#*=}"; shift ;;
        -m=*|--model_name=*) MODEL_NAME="${i#*=}"; shift ;;
        -l=*|--model_location=*) MODEL_LOCATION="${i#*=}"; shift ;;
        -i=*|--image_uri=*) IMAGE_URI="${i#*=}"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
esac
done

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
