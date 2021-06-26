"""
Camshift auf ein optical flow frame. Hat ganz gut am anfang geklappt, dann haut der frame iwann ab. 

Vlt pixel verstärken oder roi ?
"""

import numpy as np
import cv2 as cv

# VIDEOPATH = "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"
VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\DL Videos\\_tigfCJFLZg_00187.mp4'
backSub2 = cv.createBackgroundSubtractorMOG2(detectShadows=False)

#Meanshift init: 
x, y, x2, y2 = 213, 196,322, 317 # simply hardcoded the values #TODO from labelling ? 
w= abs(x-x2)
h = abs(y-y2)
track_window = (x, y, w, h)




# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

#Capture:
cap = cv.VideoCapture(VIDEOPATH)
ret, frame1 = cap.read()


prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255


# set up the ROI for tracking (camshift)
roi = frame1[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

while(1):
    ret, frame2 = cap.read()

    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 7, 1.5, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    
    #INFO
    """
    Tut ganz gut aber wir brauchen noch klarere trennung von springer und background

    """
    #Camshift
    mean_hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([mean_hsv],[0],roi_hist,[0,180],1)
    ret, track_window = cv.CamShift(dst, track_window, term_crit)


    # Draw camshift on img
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv.polylines(bgr,[pts],True, 255,2)
    cv.imshow('img2',img2)
    cv.imshow("im3",cv.polylines(frame2,[pts],True, 255,2))
    
    # Draw meanshift on img:
    # x,y,w,h = track_window
    # test_im = cv.rectangle(bgr, (x,y), (x+w,y+h), 255,2)
    # cv.imshow('img2',test_im)
    # cv.imshow("noraml",cv.rectangle(frame2, (x,y), (x+w,y+h), 255,2))


    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next