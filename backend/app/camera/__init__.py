# ~/app/camera/__init__.py
from .capture import AsyncCameraCapture
from .depthai_capture import DepthAICamera
from .stream import WebRTCStreamer, VideoStream
from typing import Union

# Maintain backward compatibility
CameraCapture = AsyncCameraCapture
CameraType = Union[AsyncCameraCapture, DepthAICamera]

__all__ = [
    'AsyncCameraCapture',
    'CameraCapture',  # Backward compatibility
    'DepthAICamera',
    'WebRTCStreamer',
    'VideoStream',
    'CameraType'
]