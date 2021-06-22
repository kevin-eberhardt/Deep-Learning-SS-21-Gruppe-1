import cv2
import numpy as np


#Color_Range for masking
low = np.array([0, 0, 130])
high = np.array([255, 145, 255])

VIDEOPATH = 'C:\\Users\\Name\\Desktop\\deep learning Projekt\\videos\\tigfCJFLZg_00154\\_tigfCJFLZg_00154_Trim.mp4'
# VIDEOPATH = 'C:\\Users\\Julius\\OneDrive - bwedu\\Studium\\06_Semester\\Deep Learning\\Videos\\_tigfCJFLZg_00146.mp4'

cap = cv2.VideoCapture(VIDEOPATH)

while True:
    ret, frame = cap.read()

    #convert to HSV
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #cv2.imshow('Masked Frame', mask)

    #mask image
    mask = cv2.inRange(hsv,low,high)
    cv2.imshow('Masked Frame', mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours,-1,(0,0,255),2)
    cv2.imshow('Frame', frame)






    if cv2.waitKey(100) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
