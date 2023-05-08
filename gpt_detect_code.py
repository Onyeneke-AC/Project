# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:50:36 2023

@author: Hi
"""

import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('explo_model.h5')

# Set up the webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("ROI Selector")

# Set initial ROI coordinates and dimensions
x, y, w, h = 200, 200, 100, 100

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw the ROI rectangle on the frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame with the ROI rectangle
    cv2.imshow("ROI Selector", frame)
    
    # Check for key press events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        # Increase ROI height
        h += 10
    elif key == ord('s'):
        # Decrease ROI height
        h -= 10
    elif key == ord('a'):
        # Decrease ROI width
        w -= 10
    elif key == ord('d'):
        # Increase ROI width
        w += 10
    elif key == ord(' '):
        # Capture the ROI image, preprocess and resize it
        roi = frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (50, 50), interpolation=cv2.INTER_AREA)
        roi_normalized = roi_resized / 255.0
        roi_input = np.reshape(roi_normalized, (1, 50, 50, 1))
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_rgb_resized = cv2.resize(roi_rgb, (50, 50), interpolation=cv2.INTER_AREA)
        roi_rgb_normalized = roi_rgb_resized / 255.0
        roi_input_rgb = np.reshape(roi_rgb_normalized, (1, 50, 50, 3))
        
        # Send the ROI image to the model and get the predicted label
        pred = model.predict(roi_input_rgb)
        label = np.argmax(pred)
        print(label)
        
        # Draw the label text on the frame
        cv2.putText(frame, f"Gesture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Update the ROI rectangle coordinates and dimensions
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)

# Release the webcam and destroy the window
cap.release()
cv2.destroyAllWindows()

