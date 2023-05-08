# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:53:33 2022

@author: Hi
"""

# Import required modules
#import os
import pandas as pd
import numpy as np
#import cv2
#import matplotlib
import tensorflow as tf
#from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, ReLU, Dropout
#from matplotlib import pyplot as plt
from Conv_jpg_dataset_to_numpy_array import X, Y, X_val, Y_val
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0)


# Remove the extra dimensions from the target tensor
#Y_train = Y_train[:, :num_classes]
#Y_test = Y_test[:, :num_classes]
with tf.device('/device:CPU:0'):
    # Initialize the model
    model = Sequential()

    # Add the convolutional and max pooling layers
    model.add(Conv2D(filters = 32,kernel_size = (7,7),strides = (1,1),input_shape = (50,50,3)))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=None, padding='valid'))
    model.add(ReLU())
    
    model.add(Dropout(0.3))
    
    model.add(Conv2D(filters = 16,kernel_size = (3,3),strides = (1,1)))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=None, padding='valid'))
    model.add(ReLU())
    
    model.add(Flatten())
    
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(20,activation='softmax'))
    
    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=1e-6),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        
    # Train the model
    history = model.fit(X_train,y_train, epochs=100, validation_data = (X_test,y_test))
    
    history.history
    
    df_loss_acc = pd.DataFrame(history.history)
    df_loss= df_loss_acc[['loss','val_loss']]
    df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
    df_acc= df_loss_acc[['accuracy','val_accuracy']]
    df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
    df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
    df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
    # Evaluate the model
    #score = model.evaluate(X_test, Y_test)
    
    y_pred = model.predict(X_val)
    
    for x in range(len(X_val)): print('Actual value-->',Y_val[x],'   Predicted value-->',np.argmax(y_pred[x]))
    
    model.save('cnn_model_keras2.h5')
    # Print test accuracy
    #print('Test accuracy:', score[1])