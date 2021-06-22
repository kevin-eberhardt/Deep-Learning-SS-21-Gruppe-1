from model_training.model.configs import YOLO_COCO_CLASSES, YOLO_INPUT_SIZE, TRAIN_CLASSES, YOLO_FRAMEWORK
from validation_video import detect_video, yolo3_detect_video_2a
from model_training.model.yolov3 import Create_Yolo

video_path = "./validation/videos/"
output_path = "./validation/video_output/"

video_file_name = "_tigfCJFLZg_00347.mp4"

file_path = video_path + video_file_name
complete_path = "/Users/kevineberhardt/Coding/Privates/Deep-Learning-SS-21-Gruppe-1/yolov3/validation/videos/_tigfCJFLZg_00347.mp4"
yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./trained_model/yolov3_custom_v2_half_data")
#detect_video(yolo, video_path = file_path, output_path = output_path + video_file_name, score_threshold=0.32, iou_threshold=0.32, CLASSES=TRAIN_CLASSES)
yolo3_detect_video_2a(video_path = complete_path, output_dir=output_path, score_threshold=0.32, iou_threshold=0.32,draw_roi=False, zoom=0.3,show_diver=True)
