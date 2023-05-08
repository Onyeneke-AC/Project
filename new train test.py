# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 12:42:53 2023

@author: Hi
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Conv_jpg_dataset_to_numpy_array import X_train, Y_train, X_test, Y_test


# Load the dataset
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Load the numpy array
#X_train = np.random.rand(18000, 50, 50, 3).astype('float32')
#X_test = np.random.rand(6000, 50, 50, 3).astype('float32')


#Resize the images
X_train = np.array([cv2.resize(img, (28,28)) for img in X_train])
X_test = np.array([cv2.resize(img, (28,28)) for img in X_test])

# Convert the images to grayscale
X_train = np.array([cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY) for img in X_train])
X_test = np.array([cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY) for img in X_test])

# Reshape the data to have 1 channel
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Reshape the data to fit the model
#X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
#X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

with tf.device('/device:CPU:0'):
    # Define the model architecture
    model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(20, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test))

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    print('\nTest accuracy:', test_acc)


