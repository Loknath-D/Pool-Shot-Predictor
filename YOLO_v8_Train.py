from ultralytics import YOLO
import os

os.chdir("D:\Work\Python 3");

#Load a model
model = YOLO("yolov8n.yaml");

#Use the model
results = model.train(data="config.yaml", epochs = 100);
