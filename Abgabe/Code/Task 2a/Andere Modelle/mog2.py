import cv2 as cv


VIDEOPATH = 'C:\\Users\\goetz\\Desktop\\_tigfCJFLZg_00187.mp4'

PATH_TEST =  "‪C:\\Users\\Julius\\Downloads\\vtest.avi"
backSub2 = cv.createBackgroundSubtractorMOG2(detectShadows=False)

# use auto backgroundRatio
# backSub2.setBackgroundRatio(0.9)


capture = cv.VideoCapture(VIDEOPATH)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask2 = backSub2.apply(frame,learningRate=0.1)

    backtorgb = cv.cvtColor(fgMask2,cv.COLOR_GRAY2RGB)

    cv.imshow('rgb grayscale mask', backtorgb)
    cv.imshow('high bg ratio', frame)
    
    keyboard = cv.waitKey(30)
    
    if keyboard == 'q' or keyboard == 27:
        break

