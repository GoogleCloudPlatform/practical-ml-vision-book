#!/bin/bash

PROJECT=$(gcloud config get-value project)
BUCKET=${PROJECT}

python ./jpeg_to_tfrecord.py \
       --all_data gs://cloud-ml-data/img/flower_photos/all_data.csv \
       --labels_file gs://cloud-ml-data/img/flower_photos/dict.txt \
       --project_id $PROJECT \
       --output_dir gs://${BUCKET}/data/flower_tfrecords
