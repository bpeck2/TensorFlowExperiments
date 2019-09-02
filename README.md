# TensorFlowExperiments

To use these files, please have a valid/compatible version of Python running, as well as TensorFlow:
https://www.tensorflow.org/install

These files are for experimenting only :)

**Weekly Status Update 9/1**
  -Added ImageAugmenation.py to train cat/dog images and make accurate predictions. Also provided dog.jpg sample image
  -(WIP) Working with Microsoft Azure Machine Learning Service Workspace to attempt to deploy the model to a Web Application for cloud       access to the model
     -Created an ACI (Azure Container Registry) of the model (packaged model and entry script)
     -Created a Web Service to host the Docker image created from the ACI
