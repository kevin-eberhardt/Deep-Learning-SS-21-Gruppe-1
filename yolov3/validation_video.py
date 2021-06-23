import cv2
import numpy as np
import datetime
import time
from model_training.model.configs import YOLO_COCO_CLASSES, YOLO_INPUT_SIZE, TRAIN_CLASSES, YOLO_FRAMEWORK
from model_training.model.utils import image_preprocess, postprocess_boxes, nms, draw_bbox
from model_training.model.yolov3 import Create_Yolo
import pandas as pd
import os
import tensorflow as tf
CLASS_INDECES = {0:"diver",1:"splash"}

#(x1, y1), (x2, y2) = (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3])
def splash_bbox_roi(splash_boxes,zoom=0,vid_shape=(640,480)): 
    """
    Zoom to reduce errors created through 


    Zoom can rach from 0 -> inf

    If zoom = 0: the box is 0% bigger 
    if zoom = 0.3: the box is 30% bigger 
    """
    y_coords = set()
    x_coords = set()

    for i in splash_boxes: 
        x_coords = x_coords.union([i[0],i[2]])
        y_coords = y_coords.union([i[1],i[3]])

    # min/max für x und y
    y_min = min(y_coords)
    y_max = max(y_coords)
    x_min = min(x_coords)
    x_max = max(x_coords)

    if zoom: 
        span_y = abs(y_max-y_min)
        span_x = abs(x_max-x_min)

        # we scale both sides for half the zoom
        zoom = zoom/2

        y_min = max(0,y_min-zoom*span_y)
        y_max = min(vid_shape[1],y_max+zoom*span_y,)
        x_min = max(0,x_min-zoom*span_x)
        x_max = min(vid_shape[0],x_max+zoom*span_x)

    return int(x_min),int(y_min),int(x_max),int(y_max)

def recolor_bw(image,splash_red=True):
    #black to white: 
    image = cv2.bitwise_not(image)
    
    if splash_red:
        lower_black = np.array([0,0,0], dtype = "uint16")
        upper_black = np.array([1,1,1], dtype = "uint16")
        black_mask = cv2.inRange(image, lower_black, upper_black)
        #black splash to red
        image[black_mask == 255] = [0, 0, 255]

    return image


### KNN B-Substraction
def detect_video_bgs(Yolo, video_path, output_path,log_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                 score_threshold=0.3, iou_threshold=0.45, rectangle_colors='',draw_roi=False, zoom = 0,show_diver=True):
    
    
    times, times_2 = [], []
    vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))  # output_path must be .mp4
    
    LOW = np.array([80, 0, 200])
    HIGH = np.array([255, 110, 255])

    log = pd.DataFrame(columns=["vis_px","vis_px_pc","total_px","total_px_pc","diff","diff_pc"])
    while True:
        _, img = vid.read()
        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        pred_bbox = Yolo.predict(image_data)
        t2 = time.time()

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        
        #Countour BGS: 
        hsv=cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        # mask image
        fgMask = cv2.inRange(hsv,LOW,HIGH)
        
        #(x1, y1), (x2, y2) = (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3])
        splash_boxes = [i for i in bboxes if CLASS_INDECES[int(i[5])] =="splash"]
        
        if splash_boxes: 
            splash_x_min,splash_y_min,splash_x_max,splash_y_max = splash_bbox_roi(splash_boxes=splash_boxes,zoom=zoom)

            
            #normal_image:
            number_of_white_pix = np.sum(fgMask == 255)
            number_total_pix = fgMask.shape[0]*fgMask.shape[1]
            print("Normal_image: Number of white pixels: {} ({}%)".format(number_of_white_pix, round((number_of_white_pix/number_total_pix)*100), 2))
            

            #splash_roi:
            splash_roi = fgMask[splash_y_min:splash_y_max, splash_x_min:splash_x_max]
            roi_number_of_white_pix = np.sum(splash_roi == 255)
            # roi_number_total_pix = splash_roi.shape[0]*splash_roi.shape[1]
            print("Roi: Number of white pixels: {} ({}%)".format(roi_number_of_white_pix, round((roi_number_of_white_pix/number_total_pix)*100), 2))
            

            pixel_diff = abs(roi_number_of_white_pix - number_of_white_pix)

            image = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB)

            
            if draw_roi:
                # image = draw_bbox(image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
                #splash_x_min,splash_y_min,splash_x_max,splash_y_max
                image = cv2.rectangle(image, (splash_x_min,splash_y_min), (splash_x_max,splash_y_max), (255, 0, 0), 2)
            
            else:
                # create mask and apply
                mask = np.zeros(image.shape[:2], dtype="uint8")
                cv2.rectangle(mask, (splash_x_min,splash_y_min), (splash_x_max,splash_y_max), 255, -1)
                masked = cv2.bitwise_and(image, image, mask=mask)

                image = masked


            #Recolor
            image = recolor_bw(image,splash_red=True)
            
            #Calcs
            vis_px_pc = round((roi_number_of_white_pix / number_total_pix) * 100, 2)
            total_px_pc = round((number_of_white_pix / number_total_pix) * 100, 2)
            diff_pc = round((roi_number_of_white_pix / number_of_white_pix) * 100, 2)
            
            image = cv2.putText(
                image,
                "Vis. PXs (roi): {} ({}%) Total wPXs: {} ({}%) Diff: {} ({}%) ".format(
                    roi_number_of_white_pix,
                    vis_px_pc,
                    number_of_white_pix,
                    total_px_pc,
                    pixel_diff,
                    diff_pc
                ),
                (0, 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.7, (0, 0, 255), 1
            )
            # Create logs:
            log = log.append({
                "vis_px": roi_number_of_white_pix,
                "vis_px_pc": vis_px_pc,
                "total_px": number_of_white_pix,
                "total_px_pc": total_px_pc,
                "diff": pixel_diff,
                "diff_pc": diff_pc
            }, ignore_index=True)

        else:
            if not show_diver: 
                #No splash and no diver should be shown.
                image = np.zeros(original_image.shape[:2], dtype="uint8")
                image = recolor_bw(image,splash_red=False)
                


            else:
                image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)


        t3 = time.time()
        times.append(t2 - t1)
        times_2.append(t3 - t1)

        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times) / len(times) * 1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2) / len(times_2) * 1000)


        # image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
        #                     (0, 0, 255), 2)

        # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
        
    log.to_csv(log_path)



### KNN B-Substraction
def detect_video_knn(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                 score_threshold=0.3, iou_threshold=0.45, rectangle_colors='',draw_roi=False, zoom = 0):
    
    #different background subtraction methods

    # backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)
    backSub = cv2.createBackgroundSubtractorKNN()
    
    #KNN 
    backSub.setDetectShadows(False)
    backSub.setDist2Threshold(13000)
    backSub.setkNNSamples(6)
    backSub.setNSamples(30)


    times, times_2 = [], []
    vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))  # output_path must be .mp4
    while True:
        _, img = vid.read()
        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        pred_bbox = Yolo.predict(image_data)


        t2 = time.time()

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        fgMask = backSub.apply(original_image,learningRate=0.9)
        
        #(x1, y1), (x2, y2) = (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3])
        splash_boxes = [i for i in bboxes if CLASS_INDECES[int(i[5])] =="splash"]
        
        if splash_boxes: 
            splash_x_min,splash_y_min,splash_x_max,splash_y_max = splash_bbox_roi(splash_boxes=splash_boxes,zoom=zoom)

            
            #normal_image:
            number_of_white_pix = np.sum(fgMask == 255)
            number_total_pix = fgMask.shape[0]*fgMask.shape[1]
            print("Normal_image: Number of white pixels: {} ({}%)".format(number_of_white_pix, round((number_of_white_pix/number_total_pix)*100), 2))
            

            #splash_roi:
            splash_roi = fgMask[splash_y_min:splash_y_max, splash_x_min:splash_x_max]
            roi_number_of_white_pix = np.sum(splash_roi == 255)
            # roi_number_total_pix = splash_roi.shape[0]*splash_roi.shape[1]
            print("Roi: Number of white pixels: {} ({}%)".format(roi_number_of_white_pix, round((roi_number_of_white_pix/number_total_pix)*100), 2))
            

            pixel_diff = abs(roi_number_of_white_pix - number_of_white_pix)

            image = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB)

            
            if draw_roi:
                # image = draw_bbox(image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
                #splash_x_min,splash_y_min,splash_x_max,splash_y_max
                image = cv2.rectangle(image, (splash_x_min,splash_y_min), (splash_x_max,splash_y_max), (255, 0, 0), 2)
            
            else:
                # create mask and apply
                mask = np.zeros(image.shape[:2], dtype="uint8")
                cv2.rectangle(mask, (splash_x_min,splash_y_min), (splash_x_max,splash_y_max), 255, -1)
                masked = cv2.bitwise_and(image, image, mask=mask)

                image = masked


            image = cv2.putText(
                image,
                "Vis. PXs (roi): {} ({}%) Total wPXs: {} ({}%) Diff: {} ({}%) ".format(
                    roi_number_of_white_pix,
                    round((roi_number_of_white_pix / number_total_pix) * 100, 2),
                    number_of_white_pix,
                    round((number_of_white_pix / number_total_pix) * 100, 2),
                    pixel_diff,
                    round((roi_number_of_white_pix / number_of_white_pix) * 100, 2)
                ),
                (0, 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.7, (0, 0, 255), 1
            )

        else:
            #TODO what todo with no splash images ?
            
            image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)

        t3 = time.time()
        times.append(t2 - t1)
        times_2.append(t3 - t1)

        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times) / len(times) * 1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2) / len(times_2) * 1000)


        # image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
        #                     (0, 0, 255), 2)

        # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()

def yolo3_detect_video_2a(video_path: str, output_dir: str,log_path:str, score_threshold: float = 0.3, iou_threshold: float = 0.3,draw_roi=False, zoom: float = 0,show_diver=True, yolo = None) -> None:
    """
    Custom function to label videos with our model
    """
    """if not os.path.isfile(video_path):
        raise FileNotFoundError("No video file found")"""


    # Create path:
    timestamp = str(datetime.datetime.now()).replace(":", "").replace(" ", "_").split(".")[0]
    video_name = os.path.split(video_path)[1].split(".")[0]
    threshold = str(score_threshold)
    output_path = "/".join([output_dir, video_name + "threshold-" + threshold + "_detected_" + timestamp + ".mp4"])
    # Detect and save
    
    detect_video_bgs(yolo, video_path=video_path, score_threshold=score_threshold, iou_threshold=iou_threshold, output_path=output_path,
                 input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0),draw_roi=draw_roi, zoom=zoom,show_diver=show_diver,log_path=log_path)


def detect_video(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                 score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    times, times_2 = [], []
    vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))  # output_path must be .mp4

    while True:
        _, img = vid.read()

        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

        t2 = time.time()

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)

        t3 = time.time()
        times.append(t2 - t1)
        times_2.append(t3 - t1)

        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times) / len(times) * 1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2) / len(times_2) * 1000)

        image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255), 2)
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()