import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math

# y = 2x + 2
xarr = np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0], dtype='float')
yarr = np.array([2.0,4.0,6.0,8.0,10.0,12.0,14.0], dtype='float')



model = tf.keras.Sequential([
        keras.layers.Dense(units=1, input_shape=[1])
        ])


model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xarr,yarr, epochs=1000)


INPUT_VAL = 2.5
threshold = model.predict([INPUT_VAL])
threshold = float(threshold)
thresholdFORM = format(threshold, ".2f")
print()
print("Initial X Input Value: " + str(INPUT_VAL))
print()
print("ML Predicted Y Value: " + str(thresholdFORM))


ax = plt.subplot(1, 1, 1)
ax.scatter(xarr, yarr)
ax.plot(INPUT_VAL, threshold, 'or')
testarr = np.array([]);
for i in range(0, math.ceil(INPUT_VAL + 1)):
    testarr = np.append(testarr, i);
    
ax.plot(model.predict(testarr), 'y', linestyle=':')
if INPUT_VAL>=xarr[len(xarr) - 1]:
    plt.axis([0.0, INPUT_VAL + 1, 0.0, threshold + 2])



plt.show()