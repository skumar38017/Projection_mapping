# ~/app/camera_factory.py
import logging
from typing import Optional, Union
import asyncio
from .camera.depthai_capture import DepthAICamera
from .camera.capture import AsyncCameraCapture

logger = logging.getLogger(__name__)

class CameraFactory:
    @staticmethod
    async def create_camera(
        use_depthai: bool = True, 
        max_retries: int = 3,
        preferred_indices: list = [0, 1, 2, '/dev/video0', '/dev/video1', '/dev/video2', '/dev/video3', '/dev/video4']
    ) -> Optional[Union[DepthAICamera, AsyncCameraCapture]]:
        """Create appropriate camera instance with fallback
        
        Args:
            use_depthai: Whether to try DepthAI camera first
            max_retries: Maximum number of retries for regular cameras
            preferred_indices: Ordered list of camera sources to try
            
        Returns:
            Initialized camera instance or None if all attempts fail
        """
        last_error = None
        
        # Try DepthAI first if requested
        if use_depthai:
            try:
                logger.info("Attempting to initialize DepthAI camera")
                camera = DepthAICamera()
                await camera.start()
                logger.info("Successfully initialized DepthAI camera")
                return camera
            except Exception as e:
                last_error = f"DepthAI: {str(e)}"
                logger.warning(f"DepthAI camera not available: {last_error}")
        
        # Try regular cameras with retries
        logger.info("Attempting to initialize regular camera")
        for attempt in range(max_retries):
            for source in preferred_indices:
                try:
                    logger.info(f"Attempt {attempt + 1}: Trying camera source {source}")
                    camera = AsyncCameraCapture(source=source)
                    await camera.start()
                    
                    # Verify we can actually get a frame
                    test_frame = await camera.get_frame(timeout=2.0)
                    if test_frame is None:
                        raise RuntimeError("Camera returned no frames")
                        
                    logger.info(f"Successfully initialized camera at source {source}")
                    return camera
                except Exception as e:
                    last_error = f"Source {source}: {str(e)}"
                    logger.warning(f"Camera initialization failed: {last_error}")
                    if camera:
                        await camera.stop()
                    await asyncio.sleep(1)  # Wait a bit before retrying
        
        logger.error(f"Failed to initialize any camera after {max_retries} attempts")
        logger.error(f"Last error: {last_error}")
        return None