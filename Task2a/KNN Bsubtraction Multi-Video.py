import cv2 as cv
import numpy as np
from time import sleep
import os
from multiprocessing import Pool

def MOG2():
    backSub = cv.createBackgroundSubtractorMOG2()
    backSub.setDetectShadows(False)
    return backSub


def KNN():
    backSub = cv.createBackgroundSubtractorKNN()
    backSub.setDetectShadows(False)
    backSub.setDist2Threshold(10000)
    backSub.setkNNSamples(8)
    backSub.setNSamples(100)
    return backSub


def do_background_subtraction(path):  
            capture = cv.VideoCapture(path)
            backSubKNN = KNN()
            backSubMOG2 = MOG2()

            while True:
                ret, frame = capture.read()
                if frame is None:
                    break

                fgMaskKNN = backSubKNN.apply(frame,learningRate=0.9)
                fgMaskMOG2 = backSubMOG2.apply(frame,learningRate=0.9)
                
                sleep(0.1)                
                
                cv.imshow(str(path[37:]) + " KNN", fgMaskKNN)
                cv.imshow(str(path[37:]) + " MOG2", fgMaskMOG2)
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
            
            
            
        


