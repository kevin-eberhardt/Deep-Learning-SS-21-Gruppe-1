import cv2
import numpy as np
import datetime
import time
from model_training.model.configs import YOLO_COCO_CLASSES, YOLO_INPUT_SIZE, TRAIN_CLASSES, YOLO_FRAMEWORK
from model_training.model.utils import image_preprocess, postprocess_boxes, nms, draw_bbox
from model_training.model.yolov3 import Create_Yolo
import os
import tensorflow as tf
CLASS_INDECES = {0:"diver",1:"splash"}

### KNN B-Substraction
def detect_video_knn(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                 score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    
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

        if [i[5] for i in bboxes if CLASS_INDECES[int(i[5])] =="splash"]:
            #TODO JOIN and USE BOUNDINGBOXES  to create roi for splash
            
            number_of_white_pix = np.sum(fgMask == 255)
            number_total_pix = fgMask.shape[0]*fgMask.shape[1]
            print("Number of white pixels: {} ({}%)".format(number_of_white_pix, round((number_of_white_pix/number_total_pix)*100), 2))
            image = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB)
            image = cv2.putText(
                image, 
                "Number of white pixels: {} ({}%)".format(
                    number_of_white_pix, 
                    round((number_of_white_pix/number_total_pix)*100), 2
                    ),
                (0, 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,(0, 0, 255), 2
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

def yolo3_detect_video_2a(video_path: str, output_dir: str, score_threshold: float = 0.3) -> None:
    """
    Custom function to label videos with our model
    """
    # Setup yolo
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    yolo.load_weights("./trained_model/yolov3_custom_v2_half_data")

    # Create path:
    timestamp = str(datetime.datetime.now()).replace(":", "").replace(" ", "_").split(".")[0]
    video_name = os.path.split(video_path)[1].split(".")[0]
    threshold = str(score_threshold)
    output_path = "/".join([output_dir, video_name + "threshold-" + threshold + "_detected_" + timestamp + ".mp4"])
    # Detect and save
    detect_video_knn(yolo, video_path=video_path, score_threshold=score_threshold, output_path=output_path,
                 input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))


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