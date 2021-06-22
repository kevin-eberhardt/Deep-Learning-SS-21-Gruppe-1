import cv2 as cv
import numpy as np
import time



# VIDEOPATH = "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"
VIDEOPATH = 'C:\\Users\\Julius\\OneDrive - bwedu\\Studium\\06_Semester\\Deep Learning\\Videos\\_tigfCJFLZg_00146.mp4'

backSub = cv.createBackgroundSubtractorMOG2(detectShadows=False)

cap = cv.VideoCapture(VIDEOPATH)

while True: 
    ret, frame = cap.read()
    if frame is None:
        break
    
    mask = backSub.apply(frame,learningRate=0.1)
    contours,_ = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)

        cv.drawContours(frame,[cnt],-1,(0,255,0),2)
    
    cv.imshow("Frame",frame)

    keyboard = cv.waitKey(30)
    
    if keyboard == 'q' or keyboard == 27:
        break
