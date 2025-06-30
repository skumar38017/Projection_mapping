# ~/app/camera/__init__.py
from .capture import CameraCapture
from .depthai_capture import DepthAICamera
from .stream import WebRTCStreamer, VideoStream
from typing import Union

CameraType = Union[CameraCapture, DepthAICamera]
__all__ = ['CameraCapture', 'DepthAICamera', 'WebRTCStreamer', 'VideoStream', 'CameraType']