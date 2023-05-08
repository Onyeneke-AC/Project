# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:38:11 2023

@author: Hi
"""

from Explo_pred_result import *
import cv2
import warnings
warnings.filterwarnings('ignore')

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


while(cap.isOpened()):
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
                cv2.drawContours(clone, [segmented + (300, 100)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
                contours, _= cv2.findContours(thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                prediction = None
                if len(contours) > 0:
                    test_image = cv2.resize(thresholded, (128, 128))
                    test_image = np.stack((test_image,) * 3, axis=-1)
                    test_image = np.expand_dims(test_image, axis=0)
                    prediction = np.argmax(model.predict(test_image), axis=-1)[0]

                if prediction is not None:
                    label_text = label[prediction]
                    cv2.putText(clone, label_text, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            cv2.imshow("Video Feed", clone)

            num_frames += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
