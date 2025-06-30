# ~/app/camera/stream.py

import os
import cv2
import numpy as np
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from fastapi import WebSocket
import json
import asyncio
from app.detection.object_detector import ObjectDetector
from app.detection.shape_analyzer import ShapeAnalyzer
from typing import Optional

class VideoStream(VideoStreamTrack):
    def __init__(self, detector: ObjectDetector, analyzer: ShapeAnalyzer):
        super().__init__()
        self.detector = detector
        self.analyzer = analyzer
        self.current_frame = None
        self.frame_lock = asyncio.Lock()
    
    async def update_frame(self, frame: np.ndarray):
        async with self.frame_lock:
            self.current_frame = frame
    
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        async with self.frame_lock:
            if self.current_frame is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                frame[:] = (0, 0, 0)  # Black frame
            else:
                frame = self.current_frame.copy()
        
        # Convert to VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

class WebRTCStreamer:
    def __init__(self, detector: ObjectDetector, analyzer: ShapeAnalyzer):
        self.detector = detector
        self.analyzer = analyzer
        self.pcs = set()
        self.video_stream = VideoStream(detector, analyzer)
    
    async def handle_connection(self, websocket: WebSocket):
        pc = RTCPeerConnection()
        self.pcs.add(pc)
        
        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            if pc.iceConnectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)
        
        # Add video stream
        pc.addTrack(self.video_stream)
        
        try:
            # Handle signaling
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message["type"] == "offer":
                    # Set remote description
                    offer = RTCSessionDescription(
                        sdp=message["sdp"],
                        type=message["type"]
                    )
                    await pc.setRemoteDescription(offer)
                    
                    # Create answer
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    
                    # Send answer
                    await websocket.send_text(json.dumps({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp
                    }))
                
                elif message["type"] == "candidate":
                    await pc.addIceCandidate(message["candidate"])
        
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            await pc.close()
            self.pcs.discard(pc)
    
    async def update_frame(self, frame: np.ndarray):
        # Process frame with detector and analyzer
        detections = self.detector.detect(frame)
        processed_frame = frame.copy()
        
        for det in detections:
            shape_info = self.analyzer.analyze(frame, det.bbox)
            if shape_info:
                # Draw bounding box and shape info
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw shape info
                info_text = f"{det.label}: {shape_info.shape_type} ({shape_info.width:.1f}x{shape_info.height:.1f}x{shape_info.depth:.1f})"
                cv2.putText(processed_frame, info_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        await self.video_stream.update_frame(processed_frame)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        return self.video_stream.current_frame