#!/bin/bash

PROJECT=$(gcloud config get-value project)
BUCKET=${PROJECT}-mlvision   # CHANGE THIS to reflect your bucket

INPUT=gs://practical-ml-vision-book-data/flowers_5_jpeg/flower_photos/all_data.csv
OUTPUT=gs://${BUCKET}/data/flowers_tftransform

## To run locally, uncomment these lines
gsutil cat $INPUT | head -100 > /tmp/top.csv
INPUT=/tmp/top.csv
OUTPUT=./flower_tftransform

# Run
echo "INPUT=$INPUT OUTPUT=$OUTPUT"
python3 -m jpeg_to_tfrecord_tft \
       --all_data $INPUT \
       --labels_file gs://practical-ml-vision-book-data/flowers_5_jpeg/flower_photos/dict.txt \
       --project_id $PROJECT \
       --output_dir $OUTPUT \
       --resize '448,448'
