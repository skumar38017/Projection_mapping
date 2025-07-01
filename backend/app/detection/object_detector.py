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
    def __init__(self, model_path: str = "yolov8x.pt"):  # Using larger x model
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        # Minimum confidence threshold
        self.conf_threshold = 0.25
        # Minimum IoU threshold for NMS
        self.iou_threshold = 0.45
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in image and return list of detections"""
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            agnostic_nms=True,  # Class-agnostic NMS
            max_det=100,  # Maximum number of detections
            classes=None,  # Detect all classes
            verbose=False  # Disable verbose output
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.class_names[cls_id]
                
                # Skip detections that are too small
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    label=label,
                    confidence=conf
                ))
        
        return detections