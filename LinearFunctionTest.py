"""

*******************************************************************************
**************************Linear Function Predictor****************************
*******************************************************************************

Created By: Brian Peck
With References from Google's TensorFlow Tutorials: https://www.tensorflow.org/tutorials


"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math

# y = 2x + 2
xarr = np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0], dtype='float')
yarr = np.array([2.0,4.0,6.0,8.0,10.0,12.0,14.0], dtype='float')


# Build the model, consists of 1 neuron analyzing 1-dimensional arrays
model = tf.keras.Sequential([
        keras.layers.Dense(units=1, input_shape=[1])
        ])

# Compile the model using a SGD Optimizer and MSE loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

# Begin training the model, set to a default 1000 iterations (epochs)
model.fit(xarr,yarr, epochs=1000)

# User Inputted Values
INPUT_VAL = np.array([10.0,15.0,20.0])

# Test Input Values against our trained model
PREDICT_VALUES = model.predict([INPUT_VAL])

# Initialize a numpy array to hold our Y predicted values 
Y_VALUE_FINAL_ARRAY = np.array([]);

# Use a counter to keep track of indexes in following for loop
counter = 0

# For each predicted Y value generated
for i in PREDICT_VALUES:
    # Convert each value into a float datatype
    thresholdCONV = float(i)
    
    # Format to remove unnnecessary decimal places
    thresholdFORM = format(thresholdCONV, ".2f")
    
    # Output to console the values that were put in and the predicted values
    print("Initial X Input Value: " + str(INPUT_VAL[counter]) + "   ML Predicted Y Value: " + str(thresholdFORM))
    print()
    
    # Increment the counter for the next array index
    counter = counter + 1
    
    # Append each value to the numpy Y values array
    Y_VALUE_FINAL_ARRAY = np.append(Y_VALUE_FINAL_ARRAY, thresholdCONV)

# Create a subplot allowing for overlapping the predicted data with the existent training data
ax = plt.subplot(1, 1, 1)

# Create a scatter plot with the existing training data
ax.scatter(xarr, yarr)

# Loop through the predicted values, plot each of these values, colored orange to be distinct from training values
for i in range(0, len(Y_VALUE_FINAL_ARRAY)):
    ax.plot(INPUT_VAL[i], Y_VALUE_FINAL_ARRAY[i], 'or')

# Initialize a numpy array to hold auto-generated test values
linearr = np.array([]);

maxXVal = np.amax(INPUT_VAL)

# Generate test values to test for efficiency of the model
for i in range(0, math.ceil(maxXVal)):
    linearr = np.append(linearr, i);

# Run a prediction on each of the tested values
plotPredict = model.predict(linearr)

# Get the max Y predicted value to ensure range in graph is suitable
maxYVal = np.amax(plotPredict)

# Plot a yellow line to show the predicted line created
ax.plot(plotPredict, 'y', linestyle=':')

# Ensure the axis constraints fall within a suitable range
plt.axis([0.0, maxXVal + 1, 0.0, maxYVal + 10])

# Set a Title and Axis Labels
plt.title("Linear Model Prediction")
plt.xlabel("Inputted X Values")
plt.ylabel("Predicted Y Values")

# Display the plot in the output
plt.show()