# How to use TPUs (Tensor Processing Units)
 
 ## TPU-accelerated notebooks
 
 You can  provision a TPU-accelerated notebook on Google's Vertex AI Platform. This script sums up the necessary gcloud commands:
 [create-tpu-deep-learning-vm.sh](https://raw.githubusercontent.com/GoogleCloudPlatform/training-data-analyst/master/courses/fast-and-lean-data-science/create-tpu-deep-learning-vm.sh)
 
 Detailed instructions below.
 
 Cloud AI Platform notebooks work with TPU and TPU pods up to the largest TPUv3-2048 pod with 2048 cores.
  
 TPUs are also available for free on [Colaboratory](https://colab.sandbox.google.com/github/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/07_Keras_Flowers_TPU_xception_fine_tuned_best.ipynb) (TPU v2-8)
 and [Kaggle](https://www.kaggle.com/mgornergoogle/five-flowers-with-keras-and-xception-on-tpu) (TPU v3-8).
 
TPU basics are [explained here](https://www.kaggle.com/docs/tpu).

## Detailed instructions for provisioning a notebook with a Cloud TPU accelerator

Please use the script [create-tpu-deep-learning-vm.sh](https://raw.githubusercontent.com/GoogleCloudPlatform/training-data-analyst/master/courses/fast-and-lean-data-science/create-tpu-deep-learning-vm.sh)
to create a Vertex AI Notebook VM along with a TPU in one go.
The script ensures that both your VM and the TPU have the same version of Tensorflow. Detailed steps:

 * Go to [Google cloud console](https://console.cloud.google.com/), create a new project with billing enabled.
 * Open cloud shell (>_ icon top right) so that you can type shell commands.
 * Get the script [create-tpu-deep-learning-vm.sh](https://raw.githubusercontent.com/GoogleCloudPlatform/training-data-analyst/master/courses/fast-and-lean-data-science/create-tpu-deep-learning-vm.sh), save it to a file, chmod u+x so that you can run it
 * Run `gcloud init` to set up your project and select a default zone that
 has TPUs. You can check TPU availability in different zones in [Google cloud console](https://console.cloud.google.com/)
 Compute Engine > TPUs > CREATE TPU NODE by playing with the zone and tpu type fields. For this
 demo, you can use an 8-core TPU or a 32-core TPU pod. Both TPU v2 and v3 will work.
 Select a zone that has v3-8, v2-32, v2-8 or v3-32 availability depending on what you want to test.
 * run the TPU and VM creation script:<br/>
 `./create-tpu-deep-learning-vm.sh choose-a-name --tpu-type v3-8`
 * You can specify a Tensorflow version with `--version=2.5.0` or use `--nightly`. Most Tensorflow versions are available for TPU but sometimes a specific major.minor version nuber is required. For example, 2.3 or 2.4.2 will work but 2.4 will not.
 * When the machines are up, go to [Google cloud console](https://console.cloud.google.com/) Vertex AI > Notebooks
 and click OPEN JUPYTERLAB in front of the VM you just created.
 * Once in Jupyter, open a terminal and clone this repository:<br/>
 `git clone https://github.com/GoogleCloudPlatform/practical-ml-vision-book.git`

You are ready to train on TPU. Any of the models in Chapet 3 and Chapter 4 support TPU training.

TPU can also be provisioned manually in the [cloud console](https://console.cloud.google.com/). Go to
Compute Engine > TPUs > CREATE TPU NODE. Use the version selector to select the same version of Tensorflow as the one in your VM.
The script does the same thing but on the command line using the two
gcloud commands for creating a VM and a TPU. It adds a couple of perks:
the VM supports Jupyter notebooks out of the box, it has the TPU_NAME environment variable set pointing to your TPU,
and it can be upgraded to tf-nightly if you need cutting edge tech: add the `--nightly` parameter when you run the script.