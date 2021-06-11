import cv2 as cv
import numpy as np
import time



# VIDEOPATH = "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"
VIDEOPATH = 'C:\\Users\\Julius\\OneDrive - bwedu\\Studium\\06_Semester\\Deep Learning\\Videos\\_tigfCJFLZg_00146.mp4'
cap = cv.VideoCapture(VIDEOPATH)


# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
x, y, x2, y2 = 213, 196,322, 317 # simply hardcoded the values #TODO from labelling ? 
w= abs(x-x2)
h = abs(y-y2)

track_window = (x, y, w, h)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt #TODO ? 
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )


#MOG2: 
backSub = cv.createBackgroundSubtractorMOG2(detectShadows=False)

roi_hist = None

while True: 
    ret, frame = cap.read()

    fgMask2 = backSub.apply(frame,learningRate=0.1)

    if roi_hist != None: 
        # set up the ROI for tracking
        roi = fgMask2[y:y+h, x:x+w]
        # hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        # mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv.calcHist([fgMask2],[0],fgMask2,[180],[0,180])
        cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

    
    if ret == True:
        # hsv = cv.cvtColor(fgMask2, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([fgMask2],[0],roi_hist,[0,180],1)
      
        # apply meanshift to get the new location
        # ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        test_im = cv.rectangle(fgMask2, (x,y), (x+w,y+h), 255,2)
        cv.imshow('img2',test_im)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break


