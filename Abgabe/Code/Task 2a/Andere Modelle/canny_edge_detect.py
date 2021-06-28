"""
Mit der Edge detection sind ganz gut die Kanten rausgekommen.

TODO: optical flow ? on canny ? 
Was noch gemacht werden muss ist background abziehen und nur "wichtige areas" umranden. 
dann sollte nur noch der spinger übrig sein. 

"""


import cv2 as cv
import numpy as np

# VIDEOPATH = "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"
VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\DL Videos\\_tigfCJFLZg_00187.mp4'
cap = cv.VideoCapture(VIDEOPATH)


def findSignificantContour(edgeImg):
    contours, hierarchy = cv.findContours(
        edgeImg,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE
    )
        # Find level 1 contours
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)
      
    # From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])
    
    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    
    return [i[0] for i in contoursWithArea]
    
    # largestContour = contoursWithArea[0][0]
    # return largestContour





while True: 
    ret, img = cap.read()
    if ret: 

        edge_img = cv.Canny(img,100,200)

        cv.imshow("Detected Edges", edge_img)

        

        contours = findSignificantContour(edge_img)
        # Draw the contour on the original image
        # contourImg = np.copy(src)
        
        for contour in contours:
            cv.drawContours(img, [contour], 0, (0, 255, 0), 2, cv.LINE_AA, maxLevel=1)
        
        cv.imshow("contours",img)


        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    
    else:
        break

    



