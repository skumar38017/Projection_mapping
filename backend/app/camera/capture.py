# ~/app/camera/capture.py
import cv2
import numpy as np
import asyncio
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

class AsyncCameraCapture:
    def __init__(self, source: Union[str, int] = 0, use_gstreamer: bool = False):
        self.source = source
        self.use_gstreamer = use_gstreamer
        self.cap = None
        self.frame_queue = asyncio.Queue(maxsize=1)
        self._running = False
        self._current_frame = None
        self._depth_placeholder = None
        self.target_width = 1920
        self.target_height = 1080
        self._capture_task = None
        self._is_capturing = False
        
    def _gstreamer_pipeline(self) -> str:
        return (
            f"v4l2src device=/dev/video{self.source} ! "
            "video/x-raw, width=1280, height=1080, framerate=30/1 ! "
            "videoconvert ! appsink"
        )
    
class AsyncCameraCapture:
    def __init__(self, source: Union[str, int] = 0, use_gstreamer: bool = False):  # Changed default source to 0
        self.source = source
        self.use_gstreamer = use_gstreamer
        self.cap = None
        self.frame_queue = asyncio.Queue(maxsize=1)
        self._running = False
        self._current_frame = None
        self._depth_placeholder = None
        self.target_width = 640  # Lower default resolution
        self.target_height = 480
        self._capture_task = None
        self._is_capturing = False
        
    async def start(self):
        """Start the async camera capture"""
        if self._is_capturing:
            logger.warning("Camera is already capturing")
            return

        try:
            # Try different camera indices if default fails
            for source in [self.source, 0, 1, 2]:
                try:
                    if self.use_gstreamer:
                        pipeline = self._gstreamer_pipeline()
                        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    else:
                        source = int(source) if str(source).isdigit() else source
                        self.cap = cv2.VideoCapture(source)
                    
                    if self.cap.isOpened():
                        break
                    self.cap.release()
                except:
                    continue
            
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Could not open any camera source")
            
            # Set properties - don't fail if these don't work
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            except:
                logger.warning("Could not set camera properties - using defaults")
            
            self._running = True
            self._is_capturing = True
            self._capture_task = asyncio.create_task(self._capture_loop())
            logger.info(f"Async camera started (source: {self.source})")
        except Exception as e:
            self._is_capturing = False
            logger.error(f"Failed to start async camera: {str(e)}")
            raise

    async def stop(self):
        """Stop the async camera capture"""
        if not self._is_capturing:
            logger.warning("Camera is not currently capturing")
            return

        self._running = False
        self._is_capturing = False
        
        try:
            if self._capture_task:
                await self._capture_task
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            logger.info("Async camera stopped")
        except Exception as e:
            logger.error(f"Error stopping camera: {str(e)}")
            raise
    
    def is_capturing(self) -> bool:
        """Check if camera is currently capturing"""
        return self._is_capturing
    
    async def _capture_loop(self):
        """Optimized async loop for capturing frames"""
        while self._running:
            try:
                # Skip queue if consumer is slow
                if self.frame_queue.full():
                    await self.frame_queue.get()
                
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                # Convert to RGB and resize if needed
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))  # Fixed smaller size
                
                # Put frame in queue without waiting
                self.frame_queue.put_nowait(frame)
                self._current_frame = frame
                
                # Non-blocking sleep
                await asyncio.sleep(0)
                
            except Exception as e:
                logger.error(f"Capture error: {str(e)}")
                await asyncio.sleep(0.1)
    
    async def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get a frame asynchronously with timeout"""
        if not self._is_capturing:
            logger.warning("Cannot get frame - camera is not capturing")
            return None

        try:
            # First try to get from current frame
            if self._current_frame is not None:
                frame = self._current_frame.copy()
                if self._is_valid_frame(frame):
                    return frame
            
            # If no valid current frame, try to get from queue
            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=timeout)
                if self._is_valid_frame(frame):
                    return frame
                return None
            except asyncio.TimeoutError:
                logger.warning("Timeout while waiting for frame")
                return None
                
        except Exception as e:
            logger.error(f"Error getting frame: {str(e)}")
            return None

    def _is_valid_frame(self, frame: np.ndarray) -> bool:
        """Helper method to validate frame"""
        return (frame is not None and 
                isinstance(frame, np.ndarray) and 
                frame.size > 0 and 
                frame.shape[0] > 0 and 
                frame.shape[1] > 0)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame synchronously"""
        if not self._is_capturing:
            return None
        return self._current_frame.copy() if self._current_frame is not None else None
    
    async def get_depth_frame(self) -> Optional[np.ndarray]:
        """Return placeholder depth frame for compatibility"""
        if not self._is_capturing:
            return None
        if self._depth_placeholder is None and self._current_frame is not None:
            h, w = self._current_frame.shape[:2]
            self._depth_placeholder = np.zeros((h, w), dtype=np.float32)
        return self._depth_placeholder
    
    def has_depth_capability(self) -> bool:
        return False
    
    def get_camera_info(self) -> Tuple[str, str]:
        return ("Webcam", "Standard webcam without depth sensing")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()