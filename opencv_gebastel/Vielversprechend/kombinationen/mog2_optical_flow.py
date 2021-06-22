"""
Hat gar nicht gut funktioniert erst mog2 dann optical. Wir brauchen mog2 ganz am ende.
optical flow scheintg besser mit farbigen Pixeln zu funktionieren
"""


import cv2 as cv
import numpy as np
import time

# https://stackoverflow.com/questions/39694694/what-is-the-difference-of-these-two-parameters-in-background-subtraction-opencv
# https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html#ab8bdfc9c318650aed53ecc836667b56a

# VIDEOPATH = 'C:\\Users\\Julius\\OneDrive - bwedu\\Studium\\06_Semester\\Deep Learning\\Videos\\_tigfCJFLZg_00146.mp4'
VIDEOPATH = "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"

PATH_TEST =  "‪C:\\Users\\Julius\\Downloads\\vtest.avi"
backSub2 = cv.createBackgroundSubtractorMOG2(detectShadows=False)


cap = cv.VideoCapture(VIDEOPATH)

triggered =False


while True:
    ret, frame = cap.read()
    if frame is None:
        break
    #TODO adjust learning rate
    # Try and and error -> 0.1
    fgMask2 = backSub2.apply(frame,learningRate=0.1)

    backtorgb = cv.cvtColor(fgMask2,cv.COLOR_GRAY2BGR)

    cv.imshow('rgb grayscale mask', backtorgb)
    cv.imshow('high bg ratio', fgMask2)
    

    if not triggered:
        frame1 = backtorgb
        prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        triggered = True
    

    # optical flow: 
    next = cv.cvtColor(backtorgb,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('after optical flow',bgr)

    prevs = next

    keyboard = cv.waitKey(30)
    
    if keyboard == 'q' or keyboard == 27:
        break

