import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

override = True
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

def create_base_model(imageSize):
    image_shape = (imageSize, imageSize, 3)
    model_base = tf.keras.applications.MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
    return model_base

def build_compile__run_model(model_base, numEpochs, trainingSet, validationSet):
    # We will use a base model for preprocessing the images
    model = keras.Sequential([
        model_base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
        
        ])

    model.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['accuracy'])


    model.summary()

    steps_per_epoch = training.n
    validation_steps = validationSet.n

    model.fit_generator(trainingSet,steps_per_epoch=steps_per_epoch ,epochs=numEpochs, workers=4, validation_data=validationSet,validation_steps=validation_steps)
    
    return model


print(tf.test.is_gpu_available())



root_dir, cats_set, dogs_set, directory_set = load_data()

size_image = 160
batch_size = 50


if (override):
    model = load_model("CatDogSampleModel_bpec.h5")


    
   
else:
        training_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        validate_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        train_generation = training_gen.flow_from_directory(directory_set[0], target_size=(size_image, size_image), batch_size=batch_size, class_mode='binary')
        validate_generation = validate_gen.flow_from_directory(directory_set[1], target_size=(size_image,size_image), batch_size=batch_size, class_mode='binary')


        model_base = create_base_model(size_image)

        model_base.trainable=False
        model_base.summary()

        model = build_compile__run_model(model_base, 3, train_generation, validate_generation)





from tensorflow.keras.preprocessing import image



img1 = image.load_img("dog.jpg", target_size=(160, 160))
img = image.img_to_array(img1)
img = img/255

img = np.expand_dims(img, axis=0)

predict = model.predict(img, steps=1)
if(predict[:,:]>0.5):
    value ='Dog :%1.2f'%(predict[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
else:
    value ='Cat :%1.2f'%(1.0-predict[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))


plt.imshow(img1)
plt.show()

