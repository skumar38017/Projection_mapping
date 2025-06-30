# ~/app/camera/capture.py

import cv2
import numpy as np
import threading
import queue
import os
from typing import Optional, Tuple

class CameraCapture:
    def __init__(self, source: str = "0", use_gstreamer: bool = False):
        self.source = source
        self.use_gstreamer = use_gstreamer
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.current_frame = None
        
    def _gstreamer_pipeline(self) -> str:
        return (
            f"v4l2src device=/dev/video{self.source} ! "
            "video/x-raw, width=1280, height=720, framerate=30/1 ! "
            "videoconvert ! appsink"
        )
    
    def start(self):
        if self.use_gstreamer:
            pipeline = self._gstreamer_pipeline()
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(int(self.source))
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()
    
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Update current frame
            self.current_frame = frame.copy()
            
            # Put frame in queue (discard old frame if queue is full)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        return self.current_frame