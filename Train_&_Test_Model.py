# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 21:25:29 2022

@author: Hi
"""
# Import required libraries
import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from Conv_jpg_dataset_to_numpy_array import X_train, Y_train, X_test, Y_test

import tensorflow as tf


from keras.utils import to_categorical
'''
num_classes = 22

Y_train = Y_train[Y_train < num_classes]
Y_test = Y_test[Y_test < num_classes]

Y_train = Y_train[:, :num_classes]
Y_test = Y_test[:, :num_classes]

Y_train = to_categorical(Y_train,num_classes = num_classes)
Y_test = to_categorical(Y_test, num_classes = num_classes)
'''
with tf.device('/device:CPU:0'): #To make the CPU run the code and not the GPU
#Initialize the model
    model = Sequential()


#Adding the 2 Convloutional and max pooling layers 
    model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(50,50,3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))
   
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))


#Adding a flattening layer
    model.add(Flatten())

#Adding a dense layer with 20 neurons (one for each hand gesture)
    model.add(Dense(20, activation='softmax'))

#Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

#Training the model
    model.fit(X_train, Y_train, batch_size = 128, epochs=10, validation_data=(X_test, Y_test))

#Evaluating the model
    score = model.evaluate(X_test, Y_test)

#Print test accuracy
    print('Test accuracy:', score[1])

