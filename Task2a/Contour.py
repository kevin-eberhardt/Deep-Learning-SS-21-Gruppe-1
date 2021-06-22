import cv2
import numpy as np
from time import sleep
import os

VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\DL Videos\\_tigfCJFLZg_00060.mp4'
#VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\DL Videos\\_tigfCJFLZg_00206.mp4'


# wie lesen wir videos ein als mp4 oder einzelne frames?

paths = []
directory = r'C:\\Users\\goetz\\Desktop\\DL Videos'
for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    cap = cv2.VideoCapture(path)
#cap = cv2.VideoCapture(VIDEOPATH)
    while True:
        ret, frame = cap.read() 
        if frame is None:
            break

        # convert to HSV
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Brightness per Frame values need to be adjusted
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1.0:
            avgBright = int(round(np.mean(hsv[:,:,2]), 2))
        else:
            avgBright = 110

        low = np.array([80, 0, 200])
        high = np.array([255, avgBright, 255])

        # mask image
        mask = cv2.inRange(hsv,low,high)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # show frames
        cv2.imshow('Masked Frame', mask)
        cv2.drawContours(frame, contours,-1,(0,0,255),2)
        cv2.imshow('Frame with Contours', frame)

        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 


