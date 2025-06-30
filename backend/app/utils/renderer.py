# ~/app/utils/renderer.py

import mediapipe as mp
import cv2
import numpy as np
from typing import Tuple, Optional
import time

class MediaPipeRenderer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            enable_segmentation=True,
            static_image_mode=False
        )
        self._last_landmarks = None
        self._last_timestamp = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process frame and return both original and processed frames"""
        try:
            current_time = int(time.time() * 1000)  # Current timestamp in milliseconds
            
            # Convert to RGB if needed
            if frame.shape[2] == 4:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            elif frame.shape[2] == 1:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = frame.copy()
            
            # Process with MediaPipe
            results = self.pose.process(frame_rgb)
            
            # Create processed frame
            processed_frame = frame_rgb.copy()
            if results.pose_landmarks:
                self._last_landmarks = results.pose_landmarks
                self.mp_drawing.draw_landmarks(
                    processed_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2)
                )
            
            # Encode both frames
            ret_orig, buf_orig = cv2.imencode('.jpg', frame_rgb, [
                int(cv2.IMWRITE_JPEG_QUALITY), 85,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
            ])
            
            ret_proc, buf_proc = cv2.imencode('.jpg', processed_frame, [
                int(cv2.IMWRITE_JPEG_QUALITY), 85,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
            ])
            
            self._last_timestamp = current_time
            return (buf_orig if ret_orig else None, 
                    buf_proc if ret_proc else None)
            
        except Exception as e:
            print(f"Rendering error: {str(e)}")
            return None, None