import cv2
import numpy as np

#Color_Range for masking -> Best values:
low = np.array([80, 0, 200])
high = np.array([255, 100, 255])

# TODO: get brightness of Video -> adjust high middle value
# 60-120

#VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\DL Videos\\Better\\_tigfCJFLZg_00187.mp4'
VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\DL Videos\\_8Vy3dlHg2w_00218.mp4'

cap = cv2.VideoCapture(VIDEOPATH)

while True:
    ret, frame = cap.read() 

    # convert to HSV
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # mask image
    mask = cv2.inRange(hsv,low,high)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # show frames
    cv2.imshow('Masked Frame', mask)
    cv2.drawContours(frame, contours,-1,(0,0,255),2)
    cv2.imshow('Frame with Contours', frame)
    
    
    # # Noise reduction (je nachdem gehen die kleinsten splashes verloren gut?)
    # for c in contours:
    #     area = cv2.contourArea(c)
    #     # remove noise by size:
    #     if area > 0:
    #         #x, y koodrinate
    #         #w, h Breite , Höhe
    #         x, y, w, h = cv2.boundingRect(c)
    #         # Boxen künstlich vergrößern:
    #         cv2.rectangle(frame, (x-5, y-5), (x-5+w+10, y-5+h+10), (255,0,0),0)
    #         cv2.drawContours(frame, c, 0,(0,0,255),0)
    #     cv2.imshow('Frame without contours', frame)
        
    #     #mask image
    #     hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #     mask = cv2.inRange(hsv,low,high)
    #     cv2.imshow('Masked Frame withourt noise', mask)

    if cv2.waitKey(100) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
