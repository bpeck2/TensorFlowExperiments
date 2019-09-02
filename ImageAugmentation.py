import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Change this value if you want to build your own model!, set to True if you want to load a sample model!
override = True

# Load the data from the provided dataset
def load_data():
    data = tf.keras.utils.get_file(origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip", fname="cats_and_dogs_filtered.zip", extract=True)
    root_dir , _ = os.path.splitext(data)

    training_dir = os.path.join(root_dir, 'train')
    validate_dir = os.path.join(root_dir, 'validation')

    cats_train = os.path.join(training_dir, 'cats')
    dogs_train = os.path.join(training_dir, 'dogs')

    cats_validate = os.path.join(validate_dir, 'cats')
    dogs_validate = os.path.join(validate_dir, 'dogs')

    cats_set = [cats_train, cats_validate]
    dogs_set = [dogs_train, dogs_validate]

    directory_set = [training_dir, validate_dir]

    return root_dir,cats_set,dogs_set,directory_set

# Create a base model based on the MobileNetV2 architecture
def create_base_model(imageSize):
    image_shape = (imageSize, imageSize, 3)
    model_base = tf.keras.applications.MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
    return model_base


def build_compile__run_model(model_base, numEpochs, trainingSet, validationSet):
    # Appending the base model to a couple Sequential type layers with Keras
    model = keras.Sequential([
        model_base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
        
        ])

    # Using a RMSProp optimizer (gradient-based) and binary_crossentropy as we are comparing two different classes
    model.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['accuracy'])


    model.summary()

    steps_per_epoch = trainingSet.n
    validation_steps = validationSet.n

    model.fit_generator(trainingSet,steps_per_epoch=steps_per_epoch ,epochs=numEpochs, workers=4, validation_data=validationSet,validation_steps=validation_steps)
    
    return model

# Gather the data
root_dir, cats_set, dogs_set, directory_set = load_data()

# Set some parameters for the image size and the batch size
size_image = 160
batch_size = 50


if (override):
    # Load the sample model
    model = load_model("CatDogSampleModel_bpec.h5")


    
   
else:
        # Rescale all images relative to the RGB value scale
        training_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        validate_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        # Gather the training and validation sets
        train_generation = training_gen.flow_from_directory(directory_set[0], target_size=(size_image, size_image), batch_size=batch_size, class_mode='binary')
        validate_generation = validate_gen.flow_from_directory(directory_set[1], target_size=(size_image,size_image), batch_size=batch_size, class_mode='binary')

        # Build the base model
        model_base = create_base_model(size_image)

        # Set training to false as we do not want the images to be trained with the base model
        model_base.trainable=False
        model_base.summary()

        # Build and run the model with a default number of epochs = 3
        model = build_compile__run_model(model_base, 3, train_generation, validate_generation)




"""
***********************************
////// Test an Image Below!! \\\\\\
***********************************
"""
from tensorflow.keras.preprocessing import image


# Insert your image on the line below!
img1 = image.load_img("dog.jpg", target_size=(160, 160))

# Convert image to numpy array, rescale on RGB scale
img = image.img_to_array(img1)
img = img/255
img = np.expand_dims(img, axis=0)


# Run the prediction on the provided image!
prediction = model.predict(img, steps=1)

# Determine whether the image is classified as a Dog or Cat and provide some percentage data
if(prediction[:,:]>0.5):
    value = "Dog confidence: "+"{:.0%}".format(prediction[0,0])
    plt.text(20, 0,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
else:
    value = "Cat confidence: "+"{:.0%}".format(1 - prediction[0,0])
    plt.text(20, 0,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))

# Show the image fed into the model, and show the rest of the prediction text
plt.imshow(img1)
plt.show()

