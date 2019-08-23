"""
*******************************************************************************
********************Basic Image (Fashion MNIST) Classifier*********************
*******************************************************************************

Using files from Keras' Fashion MNIST Dataset

@author: Brian Peck
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

"""
Class abortTraining allows for training to be stopped after a certain loss
threshold is achieved. This will save time for the user!

"""
class abortTraining(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('loss') < 0.3):
                print("Training has reached optimal loss value, stopping training to save time!")
                self.model.stop_training = True
        
        
        
# Import data from the fashion mnist dataset
fashion_dataset = keras.datasets.fashion_mnist

# Obtain training and testing images from the fashion dataset
(train_images, train_labels), (validation_images, validation_labels) = fashion_dataset.load_data()

# Just a reference to the types of categories images will be placed under
target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

########################## Pre-process Image Data #############################
# Scale pixel values in range from 0 to 1
train_images = train_images / 255.0
validation_images = validation_images / 255.0


# Reshape image sizes for consistency when building the model
train_images = train_images.reshape(60000, 28, 28, 1)
validation_images = validation_images.reshape(10000, 28, 28, 1)

###############################################################################


# Prompt user whether they want to use the already generated model
print("Would you like to use a pre-generated model? (Y or N) (~25 min training time PER epoch unless you have a TPU)")
prompt = input()
prompt = prompt.strip()

# If they do, load the file provided in the repo and begin testing validation set  
if prompt == 'Y': 
        model = load_model('fashion_trainingmodel_bpeck') # modify this path if you want to load your own file!
        model.summary()
        print("Using pre-generated model, begin testing validation set...")
else: 
    # Prompt user to save their own model if they want to save their training information
    print("Would you like to save your own model? (Y or N)")
    saveModelPrompt = input()
    saveModelPrompt = saveModelPrompt.strip()
    
    # Prompt user to name the file they want to save it as
    if saveModelPrompt == 'Y':
        print("Save As: ")
        saveFile = input()
        saveFile = saveFile.strip()
        save = True
        
    # Build the model, using convolutions (filtering) and pooling to resize as well as preserve distinct features of the images    
    model = keras.Sequential([
            
                # Convolution and pooling layers
                tf.keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                
                # Flatten the images into individual pixel values
                tf.keras.layers.Flatten(),
                
                # 128 neuron layer, suppressing any negative pixel values
                tf.keras.layers.Dense(128, activation='relu'),
                
                # 10 neuron Softmax layer to wrap it up
                tf.keras.layers.Dense(10, activation='softmax')
                ])

    # Print out a nice summary of all of the keras layers
    model.summary()

    # Compile the model using Adam optimizer and loss function SCC
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
    
    # Prepare a callback to stop training when loss is optimal
    abort = abortTraining()
    
    # Begin training the data, with 5 epochs by default (1 hour training with CPU)
    model.fit(train_images, train_labels, epochs=5, callbacks=[abort])
                
    # Save the file to the local directory if the user wants to save their model
    if save:
     model.save(saveFile)

# Get the loss from testing 10,000 validation images/labels
test_loss = model.evaluate(validation_images, validation_labels)

# Accuracy is the complement of the loss
test_acc = 1 - test_loss

# Format for neatness
test_acc = format(test_acc, ".2f")

# Print out the accuracy of the testing data with our model
print("Accuracy of the testing data: " + test_acc)