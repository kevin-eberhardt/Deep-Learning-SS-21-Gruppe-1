import cv2 as cv
import numpy as np
import time

VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\_tigfCJFLZg_00187.mp4'
#VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\slow_traffic_small.mp4'
PATH_TEST =  "‪C:\\Users\\Julius\\Downloads\\vtest.avi"
backSub = cv.createBackgroundSubtractorMOG2 (history = 500, varThreshold = 40, detectShadows = False)
backSub2 = cv.createBackgroundSubtractorKNN()


# KNN nicht gut geeignet bei bewegter Kamerafahrt

#backSub.setDist2Threshold(1000)


backSub2.setDetectShadows(False)
backSub2.setDist2Threshold(2000)
backSub2.setkNNSamples(4)
backSub2.setNSamples(50)

capture = cv.VideoCapture(VIDEOPATH)

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    

    #TODO adjust learning rate
    fgMask = backSub.apply(frame,learningRate=0.4)
    fgMask2 = backSub2.apply(frame,learningRate=0.9)
    #fgMask = backSub.apply(frame)
    #fgMask2 = backSub2.apply(frame)

    cv.imshow('2', fgMask2)
    cv.imshow('1', fgMask)
    
    keyboard = cv.waitKey(30)
    
    if keyboard == 'q' or keyboard == 27:
        break


