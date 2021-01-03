#!/usr/bin/env python
# Copyright 2020 Google Inc. Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE

class _Preprocessor:    
    def __init__(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
    
    def read_from_tfr(self, proto):
        feature_description = {
            'image': tf.io.VarLenFeature(tf.float32),
            'shape': tf.io.VarLenFeature(tf.int64),
            'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label_int': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }
        rec = tf.io.parse_single_example(
            proto, feature_description
        )
        shape = tf.sparse.to_dense(rec['shape'])
        img = tf.reshape(tf.sparse.to_dense(rec['image']), shape)
        label_int = rec['label_int']
        return img, label_int
    
    def read_from_jpegfile(self, filename):
        # same code as in 05_create_dataset/jpeg_to_tfrecord.py
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=self.IMG_CHANNELS)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img
      
    def preprocess(self, img):
        return tf.image.resize_with_pad(img, self.IMG_HEIGHT, self.IMG_WIDTH)

# most efficient way to read the data
# as determined in 07a_ingest.ipynb
# splits the files into two halves and interleaves datasets
def create_preproc_dataset(pattern, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    """
    Does interleaving, parallel calls, prefetch, batching
    Caching is not a good idea on large datasets.
    """
    preproc = _Preprocessor(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    files = [filename for filename 
             in tf.random.shuffle(tf.io.gfile.glob(pattern))]
    
    if len(files) > 1:
        print("Interleaving the reading of {} files.".format(len(files)))
        def _create_half_ds(x):
            if x == 0:
                half = files[:(len(files)//2)]
            else:
                half = files[(len(files)//2):]
            return tf.data.TFRecordDataset(half,
                                          compression_type='GZIP')
        trainds = tf.data.Dataset.range(2).interleave(
            _create_half_ds, num_parallel_calls=AUTOTUNE)
    else:
        print("WARNING! Only {} files match {}".format(len(files), pattern))
        trainds = tf.data.TFRecordDataset(files,
                                          compression_type='GZIP')
    def _preproc_img_label(img, label):
        return (preproc.preprocess(img), label)
    
    trainds = (trainds
               .map(preproc.read_from_tfr, num_parallel_calls=AUTOTUNE)
               .map(_preproc_img_label, num_parallel_calls=AUTOTUNE)
               .shuffle(200)
               .prefetch(AUTOTUNE)
              )
    return trainds

def create_preproc_image(filename, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    preproc = _Preprocessor(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    img = preproc.read_from_jpegfile(filename)
    return preproc.preprocess(img)

