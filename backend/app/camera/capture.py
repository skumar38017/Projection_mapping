# ~/app/camera/capture.py
import cv2
import numpy as np
import threading
import queue
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class CameraCapture:
    def __init__(self, source: str = "0", use_gstreamer: bool = False):
        self.source = source
        self.use_gstreamer = use_gstreamer
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.current_frame = None
        self._depth_placeholder = None
        self.target_width = 1920  # Higher target resolution
        self.target_height = 1080
        
    def _gstreamer_pipeline(self) -> str:
        return (
            f"v4l2src device=/dev/video{self.source} ! "
            f"video/x-raw, width={self.target_width}, height={self.target_height}, framerate=30/1 ! "
            "videoconvert ! appsink"
        )
    
    def start(self):
        try:
            if self.use_gstreamer:
                pipeline = self._gstreamer_pipeline()
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(int(self.source))
                # Try to set higher resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            
            # Get actual resolution
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Camera started at {actual_width}x{actual_height}")
            
            self.running = True
            self.thread = threading.Thread(
                target=self._capture_loop, 
                daemon=True,
                name="CameraCaptureThread"
            )
            self.thread.start()
            logger.info(f"Camera started (source: {self.source})")
        except Exception as e:
            logger.error(f"Failed to start camera: {str(e)}")
            raise

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        logger.info("Camera stopped")
    
    def _capture_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                self.current_frame = frame.copy()
                
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {str(e)}")
                break
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            logger.warning("Timeout while waiting for frame")
            return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        return self.current_frame
    
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Return placeholder depth frame for compatibility"""
        if self._depth_placeholder is None and self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            self._depth_placeholder = np.zeros((h, w), dtype=np.float32)
        return self._depth_placeholder
    
    def has_depth_capability(self) -> bool:
        return False
    
    def get_camera_info(self) -> Tuple[str, str]:
        return ("Webcam", "Standard webcam without depth sensing")