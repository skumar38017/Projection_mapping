# ~/app/detection/object_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, NamedTuple
import os

class Detection(NamedTuple):
    bbox: tuple  # (x1, y1, x2, y2)
    label: str
    confidence: float

class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.class_names = self.model.names
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in image and return list of detections"""
        results = self.model(image)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.class_names[cls_id]
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    label=label,
                    confidence=conf
                ))
        
        return detections