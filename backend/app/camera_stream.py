# ~/app/camera_stream.py

import cv2
import asyncio
from typing import Optional
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - falling back to CPU mode")

class Camera:
    def __init__(self, src=0, gpu=False, buffer_size=3):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source")
            
        self.gpu = gpu and GPU_AVAILABLE
        self.buffer = asyncio.Queue(maxsize=buffer_size)
        self._running = False  # Don't start capturing immediately
        
        # Try to set some basic properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        except:
            print("Warning: Couldn't set camera buffer size")

    async def _capture_frames(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                await asyncio.sleep(0.1)
                continue
            
            try:
                # Convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                await self.buffer.put(frame)
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                await asyncio.sleep(0.1)

    async def get_frame_async(self) -> Optional[np.ndarray]:
        try:
            return await asyncio.wait_for(self.buffer.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def start(self):
        if not self._running:
            self._running = True
            asyncio.create_task(self._capture_frames())

    async def stop(self):
        self._running = False
        while not self.buffer.empty():
            await self.buffer.get()

    async def release_async(self):
        await self.stop()
        self.cap.release()