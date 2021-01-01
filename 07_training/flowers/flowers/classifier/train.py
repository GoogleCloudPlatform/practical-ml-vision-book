#!/usr/bin/env python
# Copyright 2020 Google Inc. Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os, shutil
import argparse
from tensorflow.data.experimental import AUTOTUNE     

from flowers.utils.util import cleanup_dir, create_strategy
from flowers.ingest.tfrecords import *
from flowers.classifier.model import *
from flowers.utils.plots import *
from flowers.classifier.model import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS

def train_and_evaluate(strategy, opts):
    train_dataset = create_preproc_dataset(
        'gs://practical-ml-vision-book/flowers_tfr/train' + opts['pattern'],
        IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
    ).batch(opts['batch_size'])
    eval_dataset = create_preproc_dataset(
        'gs://practical-ml-vision-book/flowers_tfr/valid' + opts['pattern'],
        IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
    ).batch(opts['batch_size'])

    # checkpoint and early stopping callbacks
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(opts['outdir'], 'chkpts'),
        monitor='val_accuracy', mode='max',
        save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', mode='max',
        patience=2)
    
    # model training
    with strategy.scope():
        model = create_model(opts)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=opts['lrate']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=False),
                  metrics=['accuracy']
                 )
    print(model.summary())
    history = model.fit(train_dataset, 
                        validation_data=eval_dataset,
                        epochs=opts['num_epochs'],
                        callbacks=[model_checkpoint_cb, early_stopping_cb]
                       )
    training_plot(['loss', 'accuracy'], history, 
                  os.path.join(opts['outdir'], 'training_plot.png'))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Training parameters
    parser.add_argument(
        '--job-dir', help='Top-level output directory', required=True)
    parser.add_argument(
        '--pattern', help='Files in gs://practical-ml-vision-book/flowers_tfr/train to read',
        default='-0000[01]-*')
    parser.add_argument(
        '--num_epochs', help='How many times to iterate over training patterns',
        default=3, type=int)
    parser.add_argument(
        '--distribute', default='gpus_one_machine',
        help="""
            Has to be one of:
            * cpu
            * gpus_one_machine
            * gpus_multiple_machines
            * tpu_colab
            * tpu_caip
            * the actual name of the cloud_tpu
        """)
    parser.add_argument('--resume', dest='resume', action='store_true',
                       help="Starts from checkpoints in output directory")
    
    ## model parameters
    parser.add_argument(
        '--batch_size', help='Number of records in a batch', default=32, type=int)
    parser.add_argument(
        '--l1', help='L1 regularization', default=0., type=float)
    parser.add_argument(
        '--l2', help='L2 regularization', default=0., type=float)
    parser.add_argument(
        '--lrate', help='Adam learning rate', default=0.001, type=float)
    parser.add_argument(
        '--num_hidden', help='Number of nodes in last but one layer', default=16, type=int)

    # parse arguments, and set up outdir based on job-dir
    # job-dir is set by CAIP, but outdir is what our code wants.
    args = parser.parse_args()
    opts = args.__dict__
    opts['outdir'] = opts['job_dir']
    print("Job Parameters={}".format(opts))
    
    # able to resume
    if not opts['resume']:
        cleanup_dir(opts['outdir'])
    
    # Train, evaluate, export
    strategy = create_strategy(opts['distribute'])  # has to be first/early call in program
    model = train_and_evaluate(strategy, opts)
    export_model(model, opts['outdir'])
    