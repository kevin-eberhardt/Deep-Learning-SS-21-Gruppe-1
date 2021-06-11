"""
Camshift nach mog 2 hat auch nicht gut funktioniert. Scheint der ist besser mit Farben. 
Der denkt halt alles ist eine farbe und go.

"""

import cv2 as cv
import numpy as np
import time

# https://stackoverflow.com/questions/39694694/what-is-the-difference-of-these-two-parameters-in-background-subtraction-opencv
# https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html#ab8bdfc9c318650aed53ecc836667b56a

VIDEOPATH = 'C:\\Users\\Julius\\OneDrive - bwedu\\Studium\\06_Semester\\Deep Learning\\Videos\\_tigfCJFLZg_00146.mp4'


PATH_TEST =  "‪C:\\Users\\Julius\\Downloads\\vtest.avi"
backSub2 = cv.createBackgroundSubtractorMOG2(detectShadows=False)

# use auto backgroundRatio
# backSub2.setBackgroundRatio(0.9)


cap = cv.VideoCapture(VIDEOPATH)

# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
x, y, x2, y2 = 213, 196,322, 317 # simply hardcoded the values #TODO from labelling ? 
w= abs(x-x2)
h = abs(y-y2)

track_window = (x, y, w, h)

# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )



while True:
    ret, frame = cap.read()
    if frame is None:
        break
    

    #TODO adjust learning rate
    # Try and and error -> 0.1
    fgMask2 = backSub2.apply(frame,learningRate=0.1)
    backtorgb = cv.cvtColor(fgMask2,cv.COLOR_GRAY2RGB)

    cv.imshow('rgb grayscale mask', backtorgb)
    cv.imshow('high bg ratio', fgMask2)
    
    #camshift:
    hsv = cv.cvtColor(backtorgb, cv.COLOR_RGB2HSV)
    dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    # apply camshift to get the new location
    ret, track_window = cv.CamShift(dst, track_window, term_crit)

    # Draw it on image
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv.polylines(frame,[pts],True, 255,2)
    cv.imshow('img2',img2)




    keyboard = cv.waitKey(30)
    
    if keyboard == 'q' or keyboard == 27:
        break

