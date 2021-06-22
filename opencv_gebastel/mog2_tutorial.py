"""
Mog 2 Tutorial einfach kopiert
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

# use auto backgroundRatio
# backSub2.setBackgroundRatio(0.9)


capture = cv.VideoCapture(VIDEOPATH)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    

    #TODO adjust learning rate
    # Try and and error -> 0.1
    fgMask2 = backSub2.apply(frame,learningRate=0.1)

    backtorgb = cv.cvtColor(fgMask2,cv.COLOR_GRAY2RGB)

    cv.imshow('rgb grayscale mask', backtorgb)
    cv.imshow('high bg ratio', fgMask2)
    
    keyboard = cv.waitKey(30)
    
    if keyboard == 'q' or keyboard == 27:
        break

