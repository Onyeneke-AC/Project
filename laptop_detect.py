# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:44:08 2023

@author: Hi
"""

import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('explo_model.h5')

# Open the webcam
cap = cv2.VideoCapture(0)

# Set the font for the text overlay
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)

while True:    
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours of the hand in the binary image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (this should be the hand)
    if len(contours) > 0:
        hand_contour = max(contours, key=cv2.contourArea)

        # Create a bounding box around the hand contour
        x, y, w, h = cv2.boundingRect(hand_contour)

        # Extract the hand from the frame
        hand = frame[y:y+h, x:x+w]

        # Resize the hand to 50x50 pixels
        hand_resized = cv2.resize(hand, (64, 64))

        # Preprocess the image for the model
        hand_resized = hand_resized.astype('float32') / 255.0
        hand_resized = np.expand_dims(hand_resized, axis=0)

        # Use the model to predict the gesture
        prediction = model.predict(hand_resized)

        # Get the predicted gesture label
        gesture_label = np.argmax(prediction)

        # Overlay the gesture label on the frame
        cv2.putText(frame, str(gesture_label), (x, y), font, font_scale, font_color, 2)

    # Display the frame
    cv2.imshow('Hand Gestures', frame)
    print(gesture_label)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
