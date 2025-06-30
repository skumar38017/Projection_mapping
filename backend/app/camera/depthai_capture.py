# ~/app/camera/depthai_capture.py
import depthai as dai
import numpy as np
import cv2
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DepthAICamera:
    def __init__(self):
        self._running = False
        self.device = None
        self.rgb_queue = None
        self.depth_queue = None
        self._has_depthai = True
        
        try:
            # First check if any DepthAI device is available
            available_devices = dai.Device.getAllAvailableDevices()
            if not available_devices:
                raise RuntimeError("No DepthAI devices found")
                
            self.pipeline = self._create_pipeline()
            self.device = dai.Device(self.pipeline)
            self.rgb_queue = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
            self.depth_queue = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
            
            logger.info("Successfully initialized DepthAI camera")
        except Exception as e:
            self._has_depthai = False
            logger.warning(f"DepthAI initialization failed: {str(e)}")
            raise RuntimeError("DepthAI camera not available")

    def _create_pipeline(self) -> dai.Pipeline:
        """Create DepthAI pipeline for stereo depth calculation"""
        pipeline = dai.Pipeline()
        
        # Color camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(1280, 720)
        cam_rgb.setInterleaved(False)
        
        # Stereo depth
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        
        # Configure stereo
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # Outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        
        # Linking
        cam_rgb.preview.link(xout_rgb.input)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        stereo.depth.link(xout_depth.input)
        
        return pipeline

    def get_frame(self) -> Optional[np.ndarray]:
        """Get RGB frame"""
        if not self._running:
            return None
            
        try:
            frame = self.rgb_queue.get().getCvFrame()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error getting frame: {str(e)}")
            return None

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Get depth frame"""
        if not self._running:
            return None
            
        try:
            return self.depth_queue.get().getFrame()
        except Exception as e:
            logger.error(f"Error getting depth frame: {str(e)}")
            return None

    def start(self):
        self._running = True
        logger.info("DepthAI camera started")

    def stop(self):
        self._running = False
        logger.info("DepthAI camera stopped")

    def release(self):
        self.stop()
        if self.device:
            self.device.close()
            logger.info("DepthAI camera released")

    def has_depth_capability(self) -> bool:
        return self._has_depthai

    def get_camera_info(self) -> Tuple[str, str]:
        return ("OAK-D", "DepthAI camera with stereo depth")