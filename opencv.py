import torch
import torchvision
from glob import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

cap = cv.VideoCapture(0)
_path = 'C:/Users/Yun/Document/\python/yolov5/yolov5l.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=_path)  # local model

while(True):
    ret, cam = cap.read()
    if(ret) :
        cv.imshow('camera', cam)
        result = model(cam)
        boxes = result.pandas().xyxy[0]
        print(boxes)
        if cv.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
            break
