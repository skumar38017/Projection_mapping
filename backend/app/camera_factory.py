# ~/app/camera_factory.py
import logging
from typing import Optional, Union
from .camera import DepthAICamera, CameraCapture, CameraType

logger = logging.getLogger(__name__)

class CameraFactory:
    @staticmethod
    def create_camera(use_depthai: bool = True) -> Optional[CameraType]:
        """Create appropriate camera instance with fallback"""
        try:
            if use_depthai:
                try:
                    return DepthAICamera()
                except Exception as e:
                    logger.warning(f"DepthAI camera not available, falling back to webcam: {str(e)}")
            
            # Fallback to regular camera
            return CameraCapture()
        except Exception as e:
            logger.error(f"Failed to initialize any camera: {str(e)}")
            return None