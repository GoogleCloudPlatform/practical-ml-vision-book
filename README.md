# Practical Machine Learning for Computer Vision

<a href="https://www.amazon.com/Practical-Machine-Learning-Computer-Vision/dp/1098102363">
<img src="mlvision_book_animation.gif" height="200" /></a>

Open-sourced code from the O'Reilly book
<a href="https://www.amazon.com/Practical-Machine-Learning-Computer-Vision/dp/1098102363">
Practical Machine Learning for Computer Vision</a>
by Valliappa Lakshmanan, Martin Gorner, Ryan Gillard


** This is not an official Google product **


# Quick tour through the book

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

## 1. Setup a Vertex AI Notebook instance

### Ensure that you have GPU quota

Visit the GCP console at https://console.cloud.google.com/ and navigate to IAM & Admin | Quotas. You can also navigate to it directly by visiting https://console.cloud.google.com/google.com/iam-admin/quotas 

In the Filter, start typing Nvidia and choose NVIDIA T4 GPUs. Make sure you have a region with a limit greater than zero. If not, please request a quota increase.

Note: If you want, you can do this lab with only a CPU and not a GPU. Training will take longer. Just choose the non-GPU option in the next step.

### Navigate to Notebook creation part of GCP console

Visit the GCP console at https://console.cloud.google.com/ and navigate to Vertex AI | Notebooks. You can also navigate to it directly by visiting https://console.cloud.google.com/vertex-ai/notebooks/list/instances 

Click on +New Instance at the top of the page. Then, select the latest TensorFlow Enterprise version available with one GPU.

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

Note: This job will take about 20 minutes to complete, so we will do the next step starting from an already created dataset in the bucket gs://practical-ml-vision-book/


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

Feedback? Please file an Issue in the GitHub repo.
