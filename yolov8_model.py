from ultralytics import YOLO
import os
from utilities.utilities import get_roboflow_data

#dataset = get_roboflow_data()
model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
model.train(data=f'/Users/levihuddleston/Documents/Repos/yolo_tennis/data.yaml')  # train the model
model.export(format="torchscript")