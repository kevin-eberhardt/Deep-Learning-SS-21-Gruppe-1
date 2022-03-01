import os, datetime
from model_training.model.configs import YOLO_COCO_CLASSES, YOLO_INPUT_SIZE, TRAIN_CLASSES, YOLO_FRAMEWORK
from model_training.model.utils import load_yolo_weights, detect_image, detect_video
from model_training.model.yolov3 import Create_Yolo

def yolo3_detect_video(video_path:str,output_dir:str,score_threshold:float=0.3)->None:
    """
    Custom function to label videos with our model
    """
    #Setup yolo
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    yolo.load_weights("./trained_model/yolov3_custom_v2_half_data")
    
    #Create path:
    timestamp = str(datetime.datetime.now()).replace(":","").replace(" ","_").split(".")[0]
    video_name = os.path.split(video_path)[1].split(".")[0]
    threshold = str(score_threshold)
    output_path = "/".join([output_dir,video_name+"threshold-"+threshold+"_detected_"+timestamp+".mp4"])
    #Detect and save
    detect_video(yolo, video_path=video_path, score_threshold=score_threshold, output_path=output_path, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, 
                rectangle_colors=(255,0,0))    

    
    
complete_path = "/Users/kevineberhardt/Coding/Privates/Deep-Learning-SS-21-Gruppe-1/yolov3/validation/videos/_tigfCJFLZg_00347.mp4"
yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./trained_model/yolov3_custom_v2_half_data")
#detect_video(yolo, video_path = file_path, output_path = output_path + video_file_name, score_threshold=0.32, iou_threshold=0.32, CLASSES=TRAIN_CLASSES)


#yolo3_detect_video_2a(yolo = yolo, video_path = complete_path, output_dir=output_path, log_path=log_path,score_threshold=0.32, iou_threshold=0.32,draw_roi=False, zoom=0.3,show_diver=True)

paths = []
directory = "./validation/videos/all_videos/"
for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    paths.append(path)
    
output_path =    "./validation/video_output/all_videos_task1b/"
for file_path in paths:
    video_path = "./validation/videos/"
    filename = file_path.split("/")[-1]
    log_path = os.path.join("./validation/video_logs/all_videos/", filename.replace(".mp4", ".csv"))
    yolo3_detect_video(file_path,output_path, 0.5)