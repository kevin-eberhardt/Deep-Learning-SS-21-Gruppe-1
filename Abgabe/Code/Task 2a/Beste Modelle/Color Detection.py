import cv2
import numpy as np
from time import sleep


VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\_tigfCJFLZg_00187.mp4'


cap = cv2.VideoCapture(VIDEOPATH)

# Initiale Werte des Farbbereichs
low = np.array([80, 0, 200]) # entspricht weiß
high = np.array([120, 110, 255]) # entspricht blau


while True:
    ret, frame = cap.read() 
    if frame is None:
        break

    # Konvertiert Originalframe in HSV Farbspektrum
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    

    # Masking Image
    mask = cv2.inRange(hsv,low,high)

    # Zeichnung von Konturen zur Darstellung auf Originalframe
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # show frames
    cv2.imshow('Masked Frame', cv2.bitwise_not(mask))
    cv2.drawContours(frame, contours,-1,(0,0,255),2)
    cv2.imshow('Frame with Contours', frame)
    
    if cv2.waitKey(100) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 


