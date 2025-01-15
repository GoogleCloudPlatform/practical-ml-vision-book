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
import hypertune
from distutils.util import strtobool
from tensorflow.data.experimental import AUTOTUNE     

from flowers.utils.util import cleanup_dir, create_strategy
from flowers.ingest.tfrecords import *
from flowers.classifier.model import *
from flowers.utils.plots import *
from flowers.classifier.model import MODEL_IMG_SIZE

def train_and_evaluate(strategy, opts):
    # calculate the image dimensions given that we have to center crop
    # to achieve the model image size
    IMG_HEIGHT = IMG_WIDTH = round(MODEL_IMG_SIZE / opts['crop_ratio'])
    print('Will pad input images to {}x{}, then crop them to {}x{}'.format(
        IMG_HEIGHT, IMG_WIDTH, MODEL_IMG_SIZE, MODEL_IMG_SIZE
    ))
    IMG_CHANNELS = 3

    train_dataset = create_preproc_dataset(
        os.path.join(opts['input_topdir'], 'train' + opts['pattern']),
        IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
    ).batch(opts['batch_size'])
    eval_dataset = create_preproc_dataset(
        os.path.join(opts['input_topdir'], 'valid' + opts['pattern']),
        IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
    ).batch(opts['batch_size'])

    # if number of training examples per epoch is specified
    # repeat the training dataset indefinitely
    num_steps_per_epoch = None
    if (opts['num_training_examples'] > 0):
        train_dataset = train_dataset.repeat()
        num_steps_per_epoch = opts['num_training_examples'] // opts['batch_size']
        print("Will train for {} steps".format(num_steps_per_epoch))
    
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
        model = create_model(opts, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=opts['lrate']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=False),
                  metrics=['accuracy']
                 )
    print(model.summary())
    history = model.fit(train_dataset, 
                        validation_data=eval_dataset,
                        epochs=opts['num_epochs'],
                        steps_per_epoch=num_steps_per_epoch,
                        callbacks=[model_checkpoint_cb, early_stopping_cb]
                       )
    training_plot(['loss', 'accuracy'], history, 
                  os.path.join(opts['outdir'], 'training_plot.png'))
    
    # export the model
    export_model(model, 
                 opts['outdir'],
                 IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    # report hyperparam metric
    hpt = hypertune.HyperTune()
    accuracy = np.max(history.history['val_accuracy']) # highest encountered
    nepochs = len(history.history['val_accuracy'])
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=accuracy,
        global_step=nepochs)
    print("Reported hparam metric name=accuracy value={}".format(accuracy))
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Training parameters
    parser.add_argument(
        '--job-dir', help='Top-level output directory', required=True)
    parser.add_argument(
        '--input_topdir', help='Top-level directory of the TF Records',
        default='gs://practical-ml-vision-book-data/flowers_tfr'
    )
    parser.add_argument(
        '--pattern', help='Files in {input_topdir}/train to read',
        default='-0000[01]-*')
    parser.add_argument(
        '--num_epochs', help='How many times to iterate over training patterns',
        default=3, type=int)
    parser.add_argument(
        '--num_training_examples', 
        help='Number of examples to term as a virtual epoch. If not specified, will use actual number of examples.',
        default=-1, type=int)
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
    parser.add_argument(
        '--crop_ratio', help='Images are center-cropped to this ratio', default=0.5, type=float)
    parser.add_argument('--with_color_distort',
                        type=lambda x: bool(strtobool(x)), nargs='?', const=True, default=True,
                        help="Specify True or False. Default is True.")

    # parse arguments, and set up outdir based on job-dir
    # job-dir is set by CAIP, but outdir is what our code wants.
    args = parser.parse_args()
    opts = args.__dict__
    opts['outdir'] = opts['job_dir']
    print("Job Parameters={}".format(opts))
    
    # able to resume
    if not opts['resume']:
        cleanup_dir(os.path.join(opts['outdir'], 'chkpts'))
    
    # Train, evaluate, export
    strategy = create_strategy(opts['distribute'])  # has to be first/early call in program
    train_and_evaluate(strategy, opts)
