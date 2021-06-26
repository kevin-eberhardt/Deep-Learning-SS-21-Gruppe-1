import cv2 as cv
import numpy as np
from time import sleep

VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\DL Videos\\_tigfCJFLZg_00314.mp4'

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) # Größe der Pixel: 1 default -> keine noise reduction

# Zu viel reduction -> zu wenig splash

backSub = cv.createBackgroundSubtractorKNN()

backSub.setDetectShadows(False)
backSub.setDist2Threshold(17000)   # bester wert
backSub.setkNNSamples(5)           # bester wert
backSub.setNSamples(30)            # bester wert

capture = cv.VideoCapture(VIDEOPATH)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)#learningRate=0.4
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)

    sleep(0.1)
    cv.imshow('Original', frame)
    cv.imshow('KNN best result', fgMask)
    keyboard = cv.waitKey(30)
    
    if keyboard == 'q' or keyboard == 27:
        break


