# -*- coding: utf-8 -*-
"""
ONYENEKE ANTHONY CHIDUBEM

MATRIC NUMBER: 200353

DEPARTMENT: ELECTRICAL & ELECTRONIC ENGINEERING

CODE FUNCTION: CONVERTS TRAINING AND TESTING DATASET FROM .jpg FILES to NUMPY ARRAYS
"""

# Import required libraries
import numpy as np

import os

from keras_preprocessing.image import load_img, img_to_array


#function to get the gesture label for the training and testing data 
def get_y(filename):
    if "(" not in filename:
        return 0
    start = 1
    for each_element in filename:
        if each_element == "(":
            break
        start += 1
    end = 0
    for each_element in filename:
        if each_element == ")":
            break
        end += 1
    return int(filename[start:end]) - 2;


# Path to the directory containing the images
path1 = 'C:/Users/Hi/Documents/train_all'
path2 = 'C:/Users/Hi/Documents/test_all'


# Initialize empty lists for the training and test data
x, y, x_val, y_val = [], [], [], []


# Loop over the training images
for train_file in os.listdir('C:/Users/Hi/Documents/train_all') : 
    if "(" not in train_file:
        continue
    
    # Load the image
    img = load_img(os.path.join(path1, train_file))

    # Convert the image to a numpy array
    img_array = img_to_array(img)

    # Add the image to the training data
    x.append(img_array)

    # Add the corresponding gesture label to the training data
    y.append(get_y(train_file))


# Loop over the testing images    
for test_file in os.listdir('C:/Users/Hi/Documents/test_all') : 
    if "(" not in test_file:
        continue
    
    # Load the image
    img = load_img(os.path.join(path2, test_file))

    # Convert the image to a numpy array
    img_array = img_to_array(img)

    # Add the image to the training data
    x_val.append(img_array)

    # Add the corresponding gesture label to the training data
    y_val.append(get_y(test_file))
    

# Convert the training and test data to numpy arrays
X = np.array(x).astype(float)
Y = np.array(y)
X_val = np.array(x_val).astype(float)
Y_val = np.array(y_val)
