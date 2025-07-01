# ~/app/camera/depthai_capture.py
import depthai as dai
import numpy as np
import cv2
import asyncio
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class DepthAICamera:
    def __init__(self):
        self._running = False
        self.device = None
        self.rgb_queue = None
        self.depth_queue = None
        self._current_frame = None
        self._current_depth = None
        self._has_depthai = True
        
        try:
            available_devices = dai.Device.getAllAvailableDevices()
            if not available_devices:
                raise RuntimeError("No DepthAI devices found")
        except Exception as e:
            self._has_depthai = False
            logger.warning(f"DepthAI check failed: {str(e)}")
            raise RuntimeError("DepthAI not available")

    async def start(self):
        """Initialize and start DepthAI camera asynchronously"""
        try:
            self.pipeline = self._create_pipeline()
            self.device = dai.Device(self.pipeline)
            self.rgb_queue = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
            self.depth_queue = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
            self._running = True
            logger.info("DepthAI camera started")
        except Exception as e:
            logger.error(f"Failed to start DepthAI camera: {str(e)}")
            raise

    async def stop(self):
        """Stop DepthAI camera asynchronously"""
        self._running = False
        if self.device:
            self.device.close()
        logger.info("DepthAI camera stopped")

    async def get_frame(self) -> Optional[np.ndarray]:
        """Get RGB frame asynchronously"""
        if not self._running:
            return None
        try:
            frame = self.rgb_queue.get().getCvFrame()
            self._current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self._current_frame.copy()
        except Exception as e:
            logger.error(f"Error getting frame: {str(e)}")
            return None

    async def get_depth_frame(self) -> Optional[np.ndarray]:
        """Get depth frame asynchronously"""
        if not self._running:
            return None
        try:
            self._current_depth = self.depth_queue.get().getFrame()
            return self._current_depth.copy()
        except Exception as e:
            logger.error(f"Error getting depth frame: {str(e)}")
            return None

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

    def has_depth_capability(self) -> bool:
        return self._has_depthai

    def get_camera_info(self) -> Tuple[str, str]:
        return ("OAK-D", "DepthAI camera with stereo depth")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()