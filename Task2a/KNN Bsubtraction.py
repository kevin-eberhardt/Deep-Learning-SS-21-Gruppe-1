import cv2 as cv
import numpy as np
from time import sleep

#VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\DL Videos\\Better\\_tigfCJFLZg_00187.mp4'
VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\DL Videos\\_8Vy3dlHg2w_00218.mp4'

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) # Größe der Pixel: 1 default -> keine noise reduction

# Zu viel reduction -> zu wenig splash

backSub1 = cv.createBackgroundSubtractorKNN()
backSub2 = cv.createBackgroundSubtractorKNN()


# backSub1.setDetectShadows(False)
# backSub1.setDist2Threshold(17000) # je geringer, desto mehr weiß
# backSub1.setkNNSamples(5) # geringerer Wert -> reduzierung noise
# backSub1.setNSamples(30) # geringer -> mehr punkte weiß

backSub2.setDetectShadows(False)
backSub2.setDist2Threshold(17000)   # bester wert
backSub2.setkNNSamples(5)           # bester wert
backSub2.setNSamples(30)            # bester wert

capture = cv.VideoCapture(VIDEOPATH)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask1 = backSub1.apply(frame)#learningRate=0.4
    #fgMask1 = cv.morphologyEx(fgMask1, cv.MORPH_OPEN, kernel)

    fgMask2 = backSub2.apply(frame)#learningRate=0.4
    fgMask2 = cv.morphologyEx(fgMask2, cv.MORPH_OPEN, kernel)

    sleep(0.1)
    cv.imshow('Original', frame)
    cv.imshow('KNN ohne adjustments', fgMask1)
    cv.imshow('KNN best result', fgMask2)
    keyboard = cv.waitKey(30)
    
    if keyboard == 'q' or keyboard == 27:
        break


