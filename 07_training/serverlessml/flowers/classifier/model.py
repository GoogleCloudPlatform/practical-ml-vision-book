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
from tensorflow.data.experimental import AUTOTUNE
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

from flowers.utils.augment import *
from flowers.utils.util import cleanup_dir
from flowers.ingest.tfrecords import create_preproc_image

CLASS_NAMES = 'daisy dandelion roses sunflowers tulips'.split()
MODEL_IMG_SIZE = 224 # What mobilenet expects

def create_model(opts, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    regularizer = tf.keras.regularizers.l1_l2(opts['l1'] or 0, opts['l2'] or 0)
    
    layers = [
      tf.keras.layers.experimental.preprocessing.RandomCrop(
          height=MODEL_IMG_SIZE, width=MODEL_IMG_SIZE,
          input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
          name='random/center_crop'
      ),
      tf.keras.layers.experimental.preprocessing.RandomFlip(
          mode='horizontal',
          name='random_lr_flip/none'
      )
    ]
    
    if opts['with_color_distort']:
        layers.append(
            RandomColorDistortion(name='random_contrast_brightness/none')
        )
    
    layers += [
      hub.KerasLayer(
          "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", 
          trainable=False,
          name='mobilenet_embedding'),
      tf.keras.layers.Dense(opts['num_hidden'] or 16,
                            kernel_regularizer=regularizer, 
                            activation=tf.keras.activations.relu,
                            name='dense_hidden'),
      tf.keras.layers.Dense(len(CLASS_NAMES), 
                            kernel_regularizer=regularizer,
                            activation='softmax',
                            name='flower_prob')
    ]

    # checkpoint and early stopping callbacks
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='./chkpts',
        monitor='val_accuracy', mode='max',
        save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', mode='max',
        patience=2)
    
    # create model
    return tf.keras.Sequential(layers, name='flower_classification')

def export_model(model, outdir, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    def create_preproc_image_of_right_size(filename):
        return create_preproc_image(filename, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    @tf.function(input_signature=[tf.TensorSpec([None,], dtype=tf.string)])
    def predict_flower_type(filenames):
        input_images = tf.map_fn(
            create_preproc_image_of_right_size,
            filenames,
            fn_output_signature=tf.float32
        )
        batch_pred = model(input_images) # same as model.predict()
        top_prob = tf.math.reduce_max(batch_pred, axis=[1])
        pred_label_index = tf.math.argmax(batch_pred, axis=1)
        pred_label = tf.gather(tf.convert_to_tensor(CLASS_NAMES), pred_label_index)
        return {
            'probability': top_prob,
            'flower_type_int': pred_label_index,
            'flower_type_str': pred_label
        }
    
    outpath = os.path.join(outdir, 'flowers_model')
    cleanup_dir(outpath)
    model.save(outpath,
          signatures={
              'serving_default': predict_flower_type
          })


# ## License
# Copyright 2020 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
