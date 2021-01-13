#!/usr/bin/env python3

# Copyright 2020 Google Inc. Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

r"""

Apache Beam pipeline to create TFRecord files from JPEG files stored on GCS.
This pipeline will split the data 80:10:10,
convert the images to lie in [-1, 1] range and resize them.

Modify the constants and TF Record format as needed.

Example usage:
python3 -m jpeg_to_tfrecord_tft \
       --all_data gs://cloud-ml-data/img/flower_photos/all_data.csv \
       --labels_file gs://cloud-ml-data/img/flower_photos/dict.txt \
       --project_id $PROJECT \
       --output_dir gs://${BUCKET}/data/flower_tfrecords \
       --resize 448,448

The format of the CSV files is:
    URL-of-image,label
And the format of the labels_file is simply a list of strings one-per-line.
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import apache_beam as beam
import tensorflow as tf
import numpy as np
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tfx_bsl.public import tfxio

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 448, 448, 3
LABELS = []

def _string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def decode_image(img_bytes):
    IMG_CHANNELS = 3
    return tf.image.decode_jpeg(img_bytes, channels=IMG_CHANNELS)

def assign_record_to_split(rec):
    rnd = np.random.rand()
    if rnd < 0.8:
        return ('train', rec)
    if rnd < 0.9:
        return ('valid', rec)
    return ('test', rec)

def yield_records_for_split(x, desired_split):
    split, rec = x
    # print(split, desired_split, split == desired_split)
    if split == desired_split:
        yield rec

def write_records(OUTPUT_DIR, splits, split):
    # same 80:10:10 split
    # The flowers dataset takes about 1GB, so 20 files means 50MB each
    nshards = 16 if (split == 'train') else 2
    _ = (splits
         | 'only_{}'.format(split) >> beam.FlatMap(
             lambda x: yield_records_for_split(x, split))
         | 'write_{}'.format(split) >> beam.io.tfrecordio.WriteToTFRecord(
             os.path.join(OUTPUT_DIR, split),
             file_name_suffix='.gz', num_shards=nshards)
        )

def decode_image(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=IMG_CHANNELS)
    return img

def tft_preprocess(img_record): 
    # tft_preprocess gets a batch, but decode_jpeg can only read individual files
    img = tf.map_fn(decode_image, img_record['img_bytes'],
                    fn_output_signature=tf.float32)
    img = tf.image.convert_image_dtype(img, tf.float32) # [0,1]
    img = tf.image.resize_with_pad(img, IMG_HEIGHT, IMG_WIDTH)
    return {
        'image': img,
        'label': img_record['label'],
        'label_int': img_record['label_int']
    }

def create_input_record(filename, label):
    label_list = label.to_pylist()
    filename_list = filename.to_pylist()
    assert len(filename_list) == 1 and len(filename_list[0]) == 1
    assert len(label_list) == 1 and len(label_list[0]) == 1
    contents = tf.io.read_file(filename_list[0][0]).numpy()
    return {
        'img_bytes': contents,
        'label': label_list[0][0],
        'label_int': LABELS.index(label_list[0][0].decode())
    }

def run_main(arguments):
    global IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, LABELS
    
    JOBNAME = (
            'preprocess-images-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S'))

    PROJECT = arguments['project_id']
    OUTPUT_DIR = arguments['output_dir']

    # set RUNNER using command-line arg or based on output_dir path
    on_cloud = OUTPUT_DIR.startswith('gs://')
    if arguments['runner']:
        RUNNER = arguments['runner']
        on_cloud = (RUNNER == 'DataflowRunner')
    else:
        RUNNER = 'DataflowRunner' if on_cloud else 'DirectRunner'

    # clean-up output directory since Beam will name files 0000-of-0004 etc.
    # and this could cause confusion if earlier run has 0000-of-0005, for eg
    if on_cloud:
        try:
            subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
        except subprocess.CalledProcessError:
            pass
    else:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        os.makedirs(OUTPUT_DIR)
   
    # tf.config.run_functions_eagerly(not on_cloud)

    # read list of labels
    with tf.io.gfile.GFile(arguments['labels_file'], 'r') as f:
        LABELS = [line.rstrip() for line in f]
    print('Read in {} labels, from {} to {}'.format(
        len(LABELS), LABELS[0], LABELS[-1]))
    if len(LABELS) < 2:
        print('Require at least two labels')
        sys.exit(-1)

    # resize the input images
    ht, wd = arguments['resize'].split(',')
    IMG_HEIGHT = int(ht)
    IMG_WIDTH = int(wd)
    print("Will resize input images to {}x{}".format(IMG_HEIGHT, IMG_WIDTH))
        
    # make it repeatable
    np.random.seed(10)

    # set up Beam pipeline to convert images to TF Records
    options = {
        'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
        'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
        'job_name': JOBNAME,
        'project': PROJECT,
        'max_num_workers': 20, # autoscale up to 20
        'region': arguments['region'],
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'save_main_session': True,
        'requirements_file': 'requirements.txt'
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)

    RAW_DATA_SCHEMA = tft.tf_metadata.dataset_schema.schema_utils.schema_from_feature_spec({
            'filename': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
        })
    IMG_BYTES_METADATA = tft.tf_metadata.dataset_metadata.DatasetMetadata(
        tft.tf_metadata.dataset_schema.schema_utils.schema_from_feature_spec({
            'img_bytes': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
            'label_int': tf.io.FixedLenFeature([], tf.int64)
        })
    )
    csv_tfxio = tfxio.CsvTFXIO(file_pattern=arguments['all_data'],
                               column_names=['filename', 'label'],
                               schema=RAW_DATA_SCHEMA,
                               telemetry_descriptors=['standalone_tft'])
    with beam.Pipeline(RUNNER, options=opts) as p:
        with tft_beam.Context(temp_dir=os.path.join(OUTPUT_DIR, 'tmp', 'beam_context')):
            img_records = (p
                      | 'read_csv' >> csv_tfxio.BeamSource(batch_size=1)
                      | 'img_record' >> beam.Map(
                          lambda x: create_input_record(x[0], x[1])))

            # tf.transform preprocessing
            # note that our preprocessing is simply to resize the images
            # so there is no need to be careful to run analysis only on training data

            # Ideally, we could have done csv_tfxio.TensorAdapterConfig()
            # but here, we are processing bytes, not the filenames we read from CSV
            raw_dataset = (img_records, IMG_BYTES_METADATA)

            transformed_dataset, transform_fn = (
                raw_dataset | 'tft_img' >> tft_beam.AnalyzeAndTransformDataset(tft_preprocess)
            )
            transformed_data, transformed_metadata = transformed_dataset
            transformed_data_coder = tft.coders.ExampleProtoCoder(transformed_metadata.schema)

            # write the cropped images
            splits = (transformed_data
                      | 'create_tfr' >> beam.Map(transformed_data_coder.encode)
                      | 'assign_ds' >> beam.Map(assign_record_to_split)
                      )

            for split in ['train', 'valid', 'test']:
                write_records(OUTPUT_DIR, splits, split)

            # make sure to write out a SavedModel with the tf transforms that were carried out
            _ = (
                transform_fn | 'write_tft' >> tft_beam.WriteTransformFn(
                    os.path.join(OUTPUT_DIR, 'tft'))
            )

            if on_cloud:
                print("Submitting {} job: {}".format(RUNNER, JOBNAME))
                print("Monitor at https://console.cloud.google.com/dataflow/jobs")
            else:
                print("Running on DirectRunner. Please hold on ...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--all_data',
        # pylint: disable=line-too-long
        help=
        'Path to input.  Each line of input has two fields  image-file-name and label separated by a comma',
        required=True)
    parser.add_argument(
        '--labels_file',
        help='Path to file containing list of labels, one per line',
        required=True)
    parser.add_argument(
        '--project_id',
        help='ID (not name) of your project. Ignored by DirectRunner',
        required=True)
    parser.add_argument(
        '--runner',
        help='If omitted, uses DataFlowRunner if output_dir starts with gs://',
        default=None)
    parser.add_argument(
        '--region',
        help='Cloud Region to run in. Ignored for DirectRunner',
        default='us-central1')
    parser.add_argument(
        '--resize',
        help='Specify the img_height,img_width that you want images resized.',
        default='448,448')
    parser.add_argument(
        '--output_dir', help='Top-level directory for TF Records', required=True)

    args = parser.parse_args()
    arguments = args.__dict__
    
    run_main(arguments)