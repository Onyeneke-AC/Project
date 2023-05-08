# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 01:39:32 2023

@author: Hi
"""
# import RPi.GPIO as GPIO
# import picamera
from time import time
from shayo_test_1 import *
import cv2
import warnings
warnings.filterwarnings('ignore')

# Load the video capture
# camera = PiCamera()


## Capturing the video sequence
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

label = ['0','1','+','-','*','/','Confirm','**','%','Clear','2','3','4','5','6','7','8','9']

model = load_model('explo_model.h5')
aweight = 0.5
num_frames = 0
bg = None


def run_avg(img,aweight):
    global bg
    if bg is None:
        bg = img.copy().astype('float')
        return
    cv2.accumulateWeighted(img,bg,aweight)

def segment(img,thres=25):
    global bg
    diff = cv2.absdiff(bg.astype('uint8'),img)
    _, thresholded = cv2.threshold(diff,thres,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    else:
        segmented = max(contours,key = cv2.contourArea)
    return (thresholded,segmented)

num = 0
first_number = ""
#operator = ""
#second_number = ""

while(cap.isOpened()):
    start = time()
    ret, frame = cap.read()

    if ret ==True:
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[100:300, 300:500]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, aweight)
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                num = num + 1
                cv2.drawContours(clone, [segmented + (300, 100)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
                print("Some part of hand is detecting")
                contours, _= cv2.findContours(thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                ## "Calculator Ready" printing for 3 Secs
                if num < 90:
                    cv2.putText(clone, 'Ready to Scan Hand Gesture', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                ## Enter the first Number printing for 2 Secs
                elif num > 90 and num < 150:
                    cv2.putText(clone, 'Control the appliance using their respective hand gesture', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                ## Confirmation of first number for 2 Secs
                elif num > 200 and num < 259:
                    cv2.putText(clone, "Confirmed", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    wor = "The gesture index is " + first_number
                    cv2.putText(clone, wor, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                elif num > 299:
                    res = actuate(first_number)
                    in_line = str(res)
                    cv2.putText(clone, "The gesture displayed controls socket", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    cv2.putText(clone,in_line,(50,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
                #    sleep(5)
                elif num > 500:
                   cap.release()
                   cv2.destroyAllWindows()



                for cnt in contours:
                    if cv2.contourArea(cnt) > 5000:
                        print("Hand detecting for prediction")

                        ## Inputing the first number for 12 Secs and 2 Secs for each character
                        if num > 150 and num < 200:
                            cv2.putText(clone, first_number, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                            if num % 60 == 0:
                                pred = get_prediction(thresholded)
                                if pred != "Confirm" and pred != "Clear":
                                    first_number = first_number + pred
                                elif pred == "Clear":
                                    num = 91
                                    first_number = ""
                                else:
                                    num = 481

                        elif num > 299:
                            if num % 60 == 0:
                                pred = get_prediction(thresholded)
                                if pred == "Clear":
                                    num = 0
                                    first_number = ""
                                    #operator = ""
                                    #second_number = ""

        cv2.rectangle(clone, (300, 100), (500, 300), (0, 255, 0), 2)
        cv2.putText(clone, "Gesture Controlled Calculator", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 30, 255), 3)

        num_frames += 1
        end = time()

        cv2.imshow('frame', clone)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()


print(first_number)
print(end - start)
# Clean up the GPIO pins
# GPIO.cleanup()
cv2.waitKey()
cv2.destroyAllWindows()