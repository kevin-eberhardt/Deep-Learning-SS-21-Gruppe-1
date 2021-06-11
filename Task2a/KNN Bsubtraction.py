import cv2 as cv
import numpy as np
from time import sleep

VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\_tigfCJFLZg_00187.mp4'
PATH_TEST =  "‪C:\\Users\\Julius\\Downloads\\vtest.avi"
backSub = cv.createBackgroundSubtractorKNN()


backSub.setDetectShadows(False)
backSub.setDist2Threshold(13000)
backSub.setkNNSamples(6)
backSub.setNSamples(30)

capture = cv.VideoCapture(VIDEOPATH)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame,learningRate=0.9)

    sleep(0.1)
    cv.imshow('1', fgMask)
    
    keyboard = cv.waitKey(30)
    
    if keyboard == 'q' or keyboard == 27:
        break


