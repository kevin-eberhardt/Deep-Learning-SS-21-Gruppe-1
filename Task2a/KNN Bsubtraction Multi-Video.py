import cv2 as cv
import numpy as np
from time import sleep
import os
from multiprocessing import Pool

def do_background_subtraction(path):  
            capture = cv.VideoCapture(path)
            backSub = cv.createBackgroundSubtractorKNN()
            backSub.setDetectShadows(False)
            backSub.setDist2Threshold(13000)
            backSub.setkNNSamples(6)
            backSub.setNSamples(30)

            while True:
                ret, frame = capture.read()
                if frame is None:
                    break

                fgMask = backSub.apply(frame,learningRate=0.9)
                sleep(0.1)
                cv.imshow(str(path[37:]), fgMask)
                keyboard = cv.waitKey(30)
            
                if keyboard == 'q' or keyboard == 27:
                    break


if __name__ == '__main__':
    paths = []
    
    directory = r'C:\\Users\\goetz\\Desktop\\DL Videos'
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        paths.append(path)

    with Pool(5) as p:
        print(p.map(do_background_subtraction, paths))
            
            
            
        


