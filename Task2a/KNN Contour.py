import cv2 as cv
import numpy as np
from time import sleep

#VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\DL Videos\\Better\\_tigfCJFLZg_00187.mp4'
VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\DL Videos\\_tigfCJFLZg_00187.mp4'


#Color_Range for masking
low = np.array([80, 0, 200])
high = np.array([255, 95, 255])


# noise reduction
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) # Größe der Pixel: 1 default -> keine noise reduction

# Zu viel reduction -> zu wenig splash


backSub1 = cv.createBackgroundSubtractorKNN()
backSub1.setDetectShadows(False)
backSub1.setDist2Threshold(17000)   # bester wert
backSub1.setkNNSamples(5)           # bester wert
backSub1.setNSamples(30)            # bester wert

backSub2 = cv.createBackgroundSubtractorKNN()
backSub2.setDetectShadows(False)
backSub2.setDist2Threshold(17000)   # bester wert
backSub2.setkNNSamples(5)           # bester wert
backSub2.setNSamples(30)            # bester wert

capture = cv.VideoCapture(VIDEOPATH)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    #convert to HSV
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    #mask image
    mask = cv.inRange(hsv,low,high)
    
    # hat den eindruck als würde Bild hängenbleiben -> oder liegt an schwacher grafik des pc?
    fgMask1 = backSub1.apply(mask)
    fgMask1 = cv.morphologyEx(fgMask1, cv.MORPH_OPEN, kernel)

    fgMask2 = backSub2.apply(frame)
    fgMask2 = cv.morphologyEx(fgMask2, cv.MORPH_OPEN, kernel)


    cv.imshow('Original', frame)
    cv.imshow('Masked Frame', mask)
    cv.imshow('Color best values', fgMask1)
    cv.imshow('Color testing', fgMask2)

    
    
    
    #fgMask1 = backSub2.apply(mask)
    #fgMask1 = cv.morphologyEx(fgMask1, cv.MORPH_OPEN, kernel)

    # fgMask2 = backSub2.apply(frame)#learningRate=0.4
    # fgMask2 = cv.morphologyEx(fgMask2, cv.MORPH_OPEN, kernel)

    
    #cv.imshow('Original', frame)
    #cv.imshow('Color conversion', fgMask1)
    #cv.imshow('KNN best result', fgMask2)
    
    sleep(0.3)
    keyboard = cv.waitKey(30)
    
    if keyboard == 'q' or keyboard == 27:
        break


