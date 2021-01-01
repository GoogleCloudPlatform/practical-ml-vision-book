#!/usr/bin/env python
# Copyright 2020 Google Inc. Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import matplotlib.pylab as plt
import numpy as np
import os, shutil, tempfile, subprocess

def training_plot(metrics, history, filename):
    f, ax = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    for idx, metric in enumerate(metrics):
        ax[idx].plot(history.history[metric], ls='dashed')
        ax[idx].set_xlabel("Epochs")
        ax[idx].set_ylabel(metric)
        ax[idx].plot(history.history['val_' + metric]);
        ax[idx].legend([metric, 'val_' + metric])
    
    on_cloud = filename.startswith('gs://')
    if on_cloud:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, "out.png")
            plt.savefig(tmpfilename)
            subprocess.check_call('gsutil cp {} {}'.format(
                tmpfilename, filename).split())
    else:
        plt.savefig(filename)
    
        
