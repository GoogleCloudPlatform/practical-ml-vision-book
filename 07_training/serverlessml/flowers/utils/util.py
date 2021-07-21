#!/usr/bin/env python
# Copyright 2020 Google Inc. Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import os, shutil, subprocess
import tensorflow as tf

def cleanup_dir(OUTPUT_DIR):
    on_cloud = OUTPUT_DIR.startswith("gs://")
    if on_cloud:
        try:
            subprocess.check_call("gsutil -m rm -r {}".format(OUTPUT_DIR).split())
        except subprocess.CalledProcessError:
            pass
    else:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        os.makedirs(OUTPUT_DIR)

def create_strategy(mode):
    """
    mode has be to be one of the following:
    * cpu
    * gpus_one_machine
    * gpus_multiple_machines
    * tpu_colab
    * tpu_caip
    * the actual name of the cloud_tpu
    If you are using TPUs, this method has to be the very first thing you do.
    """
    if mode == "cpu":
        print("Using CPU.")
        return tf.distribute.OneDeviceStrategy("/cpu:0")
    elif mode == "gpus_one_machine":
        print("Using {} GPUs".format(len(tf.config.experimental.list_physical_devices("GPU"))))
        return tf.distribute.MirroredStrategy()
    elif mode == "gpus_multiple_machines":
        print("Using TFCONFIG=", os.environ["TF_CONFIG"])
        return tf.distribute.experimental.MultiWorkerMirroredStrategy()
    
    # treat as tpu
    if mode == "tpu_colab":
        tpu_name = "grpc://" + os.environ["COLAB_TPU_ADDR"]
    elif mode == "tpu_caip":
        tpu_name = None
    else:
        tpu_name = mode
        print("Using TPU: ", tpu_name)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(resolver)
    # TPUs wipe out memory, so this has to be at very start of program
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices("TPU"))
    return tf.distribute.TPUStrategy(resolver)
