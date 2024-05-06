import os
import sys
import supervision
import ultralytics
from ultralytics import YOLO
from supervision import get_video_frames_generator
from supervision import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette

HOME = os.getcwd()
# print(HOME)
VIDEO_PATH = f"{HOME}/video/DJI_0332.MP4"
# ultralytics.checks()
sys.path.append(f"{HOME}/ByteTrack")
# print("supervision.__version__:", supervision.__version__)
MODEL = 'PERSON_yolov8x.pt'
model = YOLO(MODEL)
# model.fuse()

generator = get_video_frames_generator(VIDEO_PATH)
iterator = iter(generator)
frames = next(iterator)
results = model(frames)[0]
# print(type(frames))
print(results.boxes.xyxy)
print(results.boxes.cls)
