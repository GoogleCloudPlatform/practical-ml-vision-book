# Practical Machine Learning for Computer Vision

<a href="https://www.amazon.com/Practical-Machine-Learning-Computer-Vision/dp/1098102363">
<img src="mlvision_book_animation.gif" height="200" /></a>

Open-sourced code from the O'Reilly book
<a href="https://www.amazon.com/Practical-Machine-Learning-Computer-Vision/dp/1098102363">
Practical Machine Learning for Computer Vision</a>
by Valliappa Lakshmanan, Martin Gorner, Ryan Gillard


** This is not an official Google product **


# Color images

Unfortunately, the print version of the book is not in color.
For your convenience, all the images from the book can be found in the images folder
of this repository.


# Quick tour through the book

For a full tour of the book, see Full Tour (below)

Machine learning on images is revolutionizing healthcare, manufacturing, retail, and many other sectors. Many previously difficult problems can now be solved by training machine learning models to identify objects in images. Our aim in the book Practical Machine Learning for Computer Vision was to provide intuitive explanations of the ML architectures that underpin this fast-advancing field, and to provide practical code to employ these ML models to solve practical problems involving classification, measurement, detection, segmentation, representation, generation, counting, and more.

Image classification is the “hello world” of deep learning. Therefore, this codelab also provides a practical end-to-end introduction to deep learning. It can serve as a stepping stone to other deep learning domains such as natural language processing. For more details, of course, we encourage you to read the book.

## What you’ll build
In this quick tour, you’ll build an end-to-end machine learning model from the
book’s GitHub repository for image understanding using Google Cloud Vertex AI.
We will show you how to:
* Start a Vertex AI Notebook
* Prepare the 5-flowers dataset
* Train a Transfer Learning EfficientNet model to classify flowers
* Deploy the model
* Explain its predictions
* Invoke the model from a streaming pipeline.

<b> We recommend creating a brand new GCP project to try these out. Then, delete the project when you are done, to make sure that all resources have been deleted. </b>

## 1. Setup a Vertex AI Workbench instance

### Ensure that you have GPU quota

Visit the GCP console at https://console.cloud.google.com/ and navigate to IAM & Admin | Quotas. You can also navigate to it directly by visiting https://console.cloud.google.com/google.com/iam-admin/quotas 

In the Filter, start typing Nvidia and choose NVIDIA T4 GPUs. Make sure you have a region with a limit greater than zero. If not, please request a quota increase.

Note: If you want, you can do this lab with only a CPU and not a GPU. Training will take longer. Just choose the non-GPU option in the next step.

### Navigate to Notebook creation part of GCP console

Visit the GCP console at https://console.cloud.google.com/ and navigate to Vertex AI | Workbench. You can also navigate to it directly by visiting https://console.cloud.google.com/vertex-ai/workbench 

Click on +New Instance at the top of the page. Then, select TensorFlow Enterprise 2.6 with Tesla T4.

### Create a Notebook instance

Name the instance mlvision-book-gpu

Click on the checkbox to install the Nvidia driver automatically. Make sure to check the box to install the Nvidia driver. If you missed it, delete the instance and start again. 

Click Create to accept the other defaults.

This step will take about 10 minutes. 

### Clone the book’s code repository

Click on the link to Open JupyterLab

In JupyterLab, click on the git clone button (the right-most button at the top of the left panel). 
In the textbox, type in: https://github.com/GoogleCloudPlatform/practical-ml-vision-book 
Note: An alternative way to clone the repository is to launch a Terminal and then type:
```git clone https://github.com/GoogleCloudPlatform/practical-ml-vision-book```

<b>You may encounter an out of memory error with GPU if you execute multiple notebooks with Vertex AI Notebook. To avoid it, select "Shut Down All Kernels..." from the Kernel menu before opening a new notebook.</b>

## 2. Train a Transfer Learning model

This notebook contains the core machine learning to do image classification. We will improve on this end-to-end workflow in later steps.

### Open notebook

Navigate to practical-ml-vision-book/03_image_models/03a_transfer_learning.ipynb

### Clear cells

Clear cells by selecting Edit | Clear All Outputs

### Run cells

Run cells one-by-one. Read the cell. Then, execute it by clicking Shift + Enter



## 3. Prepare ML datasets [Optional]

In this step, you will create training, validation, and test datasets that consist of data that has been prepared to make ML more efficient. The data will be written out as TensorFlow Records.

### Open notebook

Navigate to practical-ml-vision-book/05_create_dataset/05_split_tfrecord.ipynb

### Create a Cloud Storage bucket

In a separate browser window, navigate to the Storage section of the GCP console: https://console.cloud.google.com/storage/browser and create a bucket. The console will not allow you to create a bucket with a name that already exists.

The bucket should be in the same region as your notebook instance.

### Configure the Dataflow job

Skip to the bottom of the notebook and find the (last-but-one) cell that contains the line
```python -m jpeg_to_tfrecord```

Change the BUCKET setting to reflect the name of the bucket you created in the previous step. For example, you might set it to be:
```BUCKET=abc-12345```

### Run Dataflow job

Run the cell by clicking Shift + Enter

### Monitor Dataflow job

View the progress of the Dataflow job by navigating to the GCP console section for Dataflow: https://console.cloud.google.com/dataflow/jobs 
When the job completes, you will see 3 datasets created in the bucket.

Note: This job will take about 20 minutes to complete, so we will do the next step starting from an already created dataset in the bucket gs://practical-ml-vision-book-data/


## 4. Train and export a SavedModel

In this step, you will train a transfer learning model on data contained in TensorFlow Records. Then, you will export the trained model in SavedModel format to a local directory named export/flowers_model. 

### Open notebook

Navigate to practical-ml-vision-book/07_training/07c_export.ipynb

### Clear cells

Clear cells by selecting Edit | Clear All Outputs

Note: By default, this notebook trains on the complete dataset and will take about 5 minutes on a GPU, but take considerably longer on a CPU. If you are using a CPU and not a GPU, change the PATTERN_SUFFIX to process only the first (00, 01) shards and to train for only 3 epochs. The resulting model will not be very accurate but it will allow you to proceed to the next step in a reasonable amount of time. You can make this change in the first cell of the “Training” section of the notebook.

### Run cells

Run cells one-by-one. Read the cell. Then, execute it by clicking Shift + Enter


## 5. Deploy model to Vertex AI

In this step, you will deploy the model as a REST web service on Vertex AI, and try out online and batch predictions as well as streaming predictions.

### Open notebook

Navigate to practical-ml-vision-book/09_deploying/09b_rest.ipynb

### Clear cells

Clear cells by selecting Edit | Clear All Outputs

### Run cells

Run cells one-by-one. Read the cell. Then, execute it by clicking Shift + Enter


## 6. Create an ML Pipeline
In this step, you will deploy the end-to-end ML workflow as an ML Pipeline so that you can run repeatable experiments easily.

Because Vertex AI Pipeline is still in preview, you will create pipelines that run OSS Kubeflow Pipelines on GKE. 

### Launch Kubeflow Pipelines on GKE

Browse to https://console.cloud.google.com/ai-platform/pipelines/clusters and click on New Instance.

### Create GKE cluster

In the Marketplace, click Configure.

Click Create a new cluster.

Check the box to allow access to Cloud APIs

Make sure the region is correct.

Click Create cluster. This will take about 5 minutes.

Deploy Kubeflow on the cluster

Change the app instance name to mlvision-book

Click Deploy.  This will take about 5 minutes.

Note Kubeflow Host ID

In the AI Platform Pipelines section of the console (you may need to click Refresh), click on Settings and note the Kubeflow Host ID. It will be something like https://40e09ee3a33a422-dot-us-central1.pipelines.googleusercontent.com
 

### Open notebook

Navigate to practical-ml-vision-book/10_mlops/10a_mlpipeline.ipynb

### Install Kubeflow

Run the first cell to pip install kfp. 

Then, restart the kernel using the button on the ribbon at the top of the notebook.

In the second cell, change the KFPHOST variable to the hostname you noted down from the AI Platform Pipelines SDK settings.
Clear cells

Clear cells by selecting Edit | Clear All Outputs

### Run cells

Run cells one-by-one. Read the cell. Then, execute it by clicking Shift + Enter

Click on the generated Run details link.

Wait for the workflow to complete.


## Congratulations
Congratulations, you've successfully built an end-to-end machine learning model for image classification.


# Full tour of book

For a shorter exploration, see Quick Tour (above)

<b> We recommend creating a brand new GCP project to try these out. Then, delete the project when you are done, to make sure that all resources have been deleted. </b>

### 1. Ensure that you have GPU quota

Visit the GCP console at https://console.cloud.google.com/ and navigate to IAM & Admin | Quotas. You can also navigate to it directly by visiting https://console.cloud.google.com/google.com/iam-admin/quotas 

In the Filter, start typing Nvidia and choose NVIDIA T4 GPUs. Make sure you have a region with a limit greater than zero. If not, please request a quota increase.

### 2. Navigate to Vertex Workbench creation part of GCP console

Visit the GCP console at https://console.cloud.google.com/ and navigate to Vertex AI | Vertex Workbench. You can also navigate to it directly by visiting https://console.cloud.google.com/vertex-ai/workbench/

Click on +New Instance at the top of the page. Then, select the TensorFlow Enterprise 2.6 with Nvidia Tesla T4.

### 3. Create a Notebook instance

Name the instance mlvision-book-gpu

Click on the checkbox to install the Nvidia driver automatically. Make sure to check the box to install the Nvidia driver. If you missed it, delete the instance and start again. 

Click on Advanced

Change Machine Type to n1-highmem-4

Change GPU Type to Nvidia Tesla T4

Change Disk | Data Disk Type to 300 GB

Change Permission | Single User | your email address

Click Create to accept the other defaults.

This step will take about 10 minutes. 

### 4. Create a Cloud Storage bucket

Navigate to the Storage section of the GCP console: https://console.cloud.google.com/storage/browser and create a bucket. 
The console will not allow you to create a bucket with a name that already exists.
The bucket should be in the same region as your notebook instance.

### 5. Clone the book’s code repository

Go to the Vertex Workbench section of the GCP console.
Click on the link to Open JupyterLab

In JupyterLab, click on the git clone button (the right-most button at the top of the left panel). 
In the textbox, type in: https://github.com/GoogleCloudPlatform/practical-ml-vision-book 
Note: An alternative way to clone the repository is to launch a Terminal and then type:
```git clone https://github.com/GoogleCloudPlatform/practical-ml-vision-book```

### 6. Run through the notebooks

* In JupyterLab, navigate to the folder practical-ml-vision-book/02_ml_models
* Open the notebook 02a.  
  * Edit | Clear All Outputs
  * Read and run each cell one-by-one by typing Shift + Enter. (or click Run | Restart Kernel and Run All Cells) 
  * Go to the list of running Terminals and Kernels (the second button from the top on the extreme left of JupyterLab)
  * Stop the 02a notebook.  <b>Stop the Kernel every time you finish running a notebook.</b> Otherwise, you will run out of memory.
* Now, open and run notebook 02b, and repeat steps listed above.
* In Chapter 3, run *only* the flowers5 notebooks (3a and 3b on MobileNet).
  * Run 3a_transfer_learning
  * Run 3b_finetune_MOBILENETV2_flowers5 -- note that if AdamW is not found, you may have to restart the kernel. See instructions in notebook.
  * Many of the flowers104 notebooks will require a more powerful machine. We did these notebooks using TPUs. See README_TPU.md for details. You can try adding more GPUs if you don't have access to TPUs but this has not been tested.
* In Chapter 4
  * Unet segmentation will work on a T4.
  * *Works in TensorFlow 2.7+ only* Follow the Readme directions in the directory to try out RetinaNet. You'll need a high-bandwidth internet connection to download and upload the 12 GB dataset. Also, you need to create a new Workbench instance with TensorFlow 2.7+ (not 2.6).

* In Chapter 5, you can run the notebooks in any order.
* In Chapter 6:
  * Change the BUCKET variable in run_dataflow.sh
  * Run the notebooks in order.
  * 6h, the TF Transform notebook, is broken (most likely a Python dependency problem)
  
* In Chapter 7, run the notebooks in order.
  * In 7c, make sure to change the BUCKET where marked.
* In Chapter 9, run the notebooks in order. 
  * Make sure to change ENDPOINT_ID, BUCKET, etc. to reflect your environment.
* In Chapter 10:
  * Start a Kubeflow Pipelines Cluster by visiting https://console.cloud.google.com/marketplace/product/google-cloud-ai-platform/kubeflow-pipelines
  * Make sure to allow access to Cloud Platform APIs from the cluster
  * Once cluster has been started, click "Deploy"
  * Once deployed, click on the link to go to the Kubeflow Pipelines console and look at the Settings
  * Note the HOST string passed
  * In JupyterLab, edit the KFPHOST variable in 10a to reflect the cluster that you just started
  * Run 10a and 10b
* In Chapter 11, run the notebooks in order.
* In Chapter 12, run the notebooks in order.

### Common issues that readers run into
* <b>Out of memory.</b> Make sure that you have shut down all previous notebooks.
* <b>AdamW not found.</b> Make sure that you restart the kernel when you start the notebook. AdamW has to be imported before first TensorFlow call.
* <b>Bucket permissions problem</b> Make sure to change BUCKET variable to something you own.
* <b>Weird GPU errors</b> Most likely, the GPU is out of memory. Please shut down other notebooks, restart kernel, and try again.


Feedback? Please file an Issue in the GitHub repo.
