# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 00:55:33 2023

@author: Hi
"""
# import RPi.GPIO as GPIO
import cv2
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import img_to_array

label = ['0','1','+','-','*','/','Confirm','**','%','Clear','2','3','4','5','6','7','8','9']
#label = ['0','1','2','3','4','5','6','7','8','9','+','-','*','/','Confirm','**','%','Clear']

## Loading model using the loacation where the model is saved
model = load_model('C:/Users/Hi/Desktop/anthony/Final Year Project/Spyder_codes/explo_model.h5')

# set the GPIO mode
# GPIO.setmode(GPIO.BCM)

'''
# Set the pins that are to be controlled as outputs
GPIO.setup(6, GPIO.OUT)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(16, GPIO.OUT)
'''
def get_prediction(img):
    for_pred = cv2.resize(img,(64,64))
    x = img_to_array(for_pred)
    x = x/255.0
    x = x.reshape((1,) + x.shape)
    gesture = str(label[np.argmax(model.predict(x))])
    return gesture

#actuate_dict = {"0" : on_1, "1" : off_1, "2" : on_2, "3" : off_2, "4" : on_3, "5" : off_3, "6" : on_4, "7" : off_4}

''' def off_1():
       GPIO.output(6, LOW)
    
    def off_2():
       GPIO.output(12, LOW)
   
    def off_3():
       GPIO.output(13, LOW)
   
    def off_4():
       GPIO.output(16, LOW)
       
    def on_1():
        GPIO.output(6, HIGH)
        
    def on_2():
        GPIO.output(12, HIGH)
        
    def on_3():
        GPIO.output(13, HIGH)
        
    def on_4():
        GPIO.output(16, HIGH)'''
    
def actuate(operator):
	# actuate_dict[operator]()
    return operator
	