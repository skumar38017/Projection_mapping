# ~/app/detection/shape_analyzer.py

import cv2
import numpy as np
from typing import Optional, Tuple
import mediapipe as mp
from app.schemas.detection import Shape3D
from .object_detector import Detection 

class ShapeAnalyzer:
    def __init__(self):
        # Initialize MediaPipe solutions with more generic settings
        self.mp_objectron = mp.solutions.objectron
        self.objectron = self.mp_objectron.Objectron(
            static_image_mode=False,
            max_num_objects=100,
            min_detection_confidence=0.4,  # Lower threshold for more detections
            min_tracking_confidence=0.3,
            model_name='Shoe'  # More generic model than 'Cup'
        )
        
        # Camera parameters (should be calibrated for your camera)
        self.focal_length = 1000  # Approximate focal length in pixels
        self.known_widths = {
            # Add more known dimensions for common objects
            # "cup": 0.08,
            # "bottle": 0.07,
            # "person": 0.5,
            # "chair": 0.4,
            # "car": 1.8,
            # "book": 0.2,
            # "cell phone": 0.07,
            # "keyboard": 0.35,
            # # Default fallback
            "default": 0.2
        }
    
    def _analyze_without_depth(self, frame: np.ndarray, detection: Detection) -> Optional[Shape3D]:
        """Analyze shape when depth information is not available"""
        x1, y1, x2, y2 = detection.bbox
        roi = frame[y1:y2, x1:x2]
        
        # Use contour-based analysis as fallback
        shape = self._contour_based_shape_detection(roi)
    
        if shape:
            # Adjust dimensions based on detection confidence and known widths
            known_width = self.known_widths.get(detection.label.lower(), self.known_widths["default"])
            scale_factor = known_width / shape.width
            shape.width = known_width
            shape.height *= scale_factor
            shape.depth *= scale_factor
    
        return shape

    def analyze(self, frame: np.ndarray, depth: Optional[np.ndarray], detection: Detection) -> Optional[Shape3D]:
        """Analyze the shape within the bounding box and estimate 3D dimensions"""
        if depth is None:
            return self._analyze_without_depth(frame, detection)
        
        x1, y1, x2, y2 = detection.bbox
        roi = frame[y1:y2, x1:x2]
        
        # Convert to RGB for MediaPipe
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Objectron
        results = self.objectron.process(roi_rgb)
        
        if results.detected_objects:
            # Get the first detected object
            detected_object = results.detected_objects[0]
            
            # Get 3D bounding box points (in normalized coordinates)
            landmarks = detected_object.landmarks_3d.landmark
            
            # Convert to pixel coordinates
            h, w = roi.shape[:2]
            points_3d = []
            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points_3d.append((x, y))
            
            if len(points_3d) >= 8:  # Objectron returns 8 points for 3D bbox
                # Calculate dimensions
                width_px = self._distance(points_3d[0], points_3d[1])
                height_px = self._distance(points_3d[0], points_3d[2])
                depth_px = self._distance(points_3d[0], points_3d[4])
                
                # Estimate real-world dimensions using depth information
                z = np.median(depth[y1:y2, x1:x2][depth[y1:y2, x1:x2] > 0])
                
                width_m = (width_px * z) / self.focal_length
                height_m = (height_px * z) / self.focal_length
                depth_m = (depth_px * z) / self.focal_length
                
                # Determine shape type based on proportions
                shape_type = self._determine_shape_type(width_m, height_m, depth_m)
                
                return Shape3D(
                    shape_type=shape_type,
                    width=width_m,
                    height=height_m,
                    depth=depth_m,
                    vertices_3d=points_3d
                )
        
        # Fallback to contour-based 2D shape detection if 3D fails
        return self._contour_based_shape_detection(roi)

    def _estimate_distance(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> float:
        """Estimate distance to object based on bounding box size (very rough approximation)"""
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Assume the object is roughly filling the bounding box
        avg_size = (bbox_width + bbox_height) / 2
        
        # This is a simplified calculation - in practice you'd need camera calibration
        # and known object sizes or depth information
        distance = (self.focal_length * 0.2) / avg_size  # 0.2m is an arbitrary reference size
        
        return max(distance, 0.1)  # Don't return less than 10cm
    
    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def _determine_shape_type(self, width: float, height: float, depth: float) -> str:
        """Determine shape type based on proportions"""
        ratios = sorted([width/height, width/depth, height/depth])
        
        if all(0.9 < r < 1.1 for r in ratios):
            return "cube"
        elif ratios[2] > 2.5:
            return "cylinder"
        elif height > width * 1.5 and height > depth * 1.5:
            return "tall"
        elif width > height * 1.5 and width > depth * 1.5:
            return "wide"
        elif depth < 0.1 * width or depth < 0.1 * height:
            return "flat"
        else:
            return "unknown"
    
    def _contour_based_shape_detection(self, roi: np.ndarray) -> Optional[Shape3D]:
        """Fallback 2D shape detection using contours"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Determine shape
        shape_type = "unknown"
        if len(approx) == 3:
            shape_type = "triangle"
        elif len(approx) == 4:
            # Get aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            shape_type = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape_type = "pentagon"
        elif len(approx) == 6:
            shape_type = "hexagon"
        elif len(approx) > 10:
            # Check if it's a circle
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            shape_type = "circle" if circularity > 0.8 else "oval"
        
        # Estimate dimensions (2D only)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return Shape3D(
            shape_type=shape_type,
            width=w,
            height=h,
            depth=0.1 * min(w, h),  # Very rough depth estimate
            vertices_3d=[]
        )