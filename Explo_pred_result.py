# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 02:25:47 2023

@author: Hi
"""

import cv2
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import img_to_array

label = ['0','1','+','-','*','/','Confirm','**','%','Clear','2','3','4','5','6','7','8','9']
#label = ['0','1','2','3','4','5','6','7','8','9','+','-','*','/','Confirm','**','%','Clear']

## Loading model using the loacation where the model is saved
model = load_model('C:/Users/Hi/Desktop/anthony/Final Year Project/Spyder_codes/explo_model.h5')



def get_prediction(img):
    for_pred = cv2.resize(img,(64,64))
    x = img_to_array(for_pred)
    x = x/255.0
    x = x.reshape((1,) + x.shape)
    pred = str(label[np.argmax(model.predict(x))])
    return pred

def get_result(first_number,operator,second_number):
	a = int(first_number)
	b = int(second_number)
	if operator == "*":
		res = a * b
	elif operator == "/":
		res = a/b
	elif operator == "+":
		res = a + b
	elif operator == "%":
		res = a % b
	elif operator == "-":
		res = a - b
	elif operator == "**":
		res = a**b
	else:
		print("Not a valid operator")
	return res