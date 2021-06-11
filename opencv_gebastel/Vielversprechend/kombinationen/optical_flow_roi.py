"""
Experimental static roi and optical flow in it. 
Worked somehow. But no ideas how to proceed with this yet.

"""


import numpy as np
import cv2 as cv

VIDEOPATH = "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"
# VIDEOPATH = 'C:\\Users\\Julius\\OneDrive - bwedu\\Studium\\06_Semester\\Deep Learning\\Videos\\_tigfCJFLZg_00146.mp4'



cap = cv.VideoCapture(VIDEOPATH)
ret, frame1 = cap.read()

#global roi
global_roi = frame1[160:320,240:400]
frame1 = global_roi
cv.imshow("roi_g",global_roi)


prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()

    global_roi = frame2[160:320,240:400]
    frame2 = global_roi
    cv.imshow("roi_g",global_roi) 
    
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 7, 1.5, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    

    cv.imshow('frame2',bgr)
    #TODO all colored white ?

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next