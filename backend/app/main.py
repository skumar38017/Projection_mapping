# ~/app/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from app.detection.object_detector import ObjectDetector
from app.detection.shape_analyzer import ShapeAnalyzer
from app.camera.stream import WebRTCStreamer
from app.schemas.detection import DetectionResult
from typing import List, Optional, Tuple
import uvloop
import asyncio
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from app.camera_stream import Camera
from app.utils.renderer import MediaPipeRenderer
import json
import time
import cv2
import numpy as np

# Install uvloop for better async performance
uvloop.install()

app = FastAPI(
    title="3D Shape Detection API",
    description="Advanced 3D shape detection with real-time streaming",
    version="1.0.0"
)

# Configure static files
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

if not STATIC_DIR.exists():
    raise RuntimeError(f"Static directory not found at {STATIC_DIR}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
executor = ThreadPoolExecutor(max_workers=4)

# Initialize components
detector = ObjectDetector()
analyzer = ShapeAnalyzer()
streamer = WebRTCStreamer(detector, analyzer)
camera = None
renderer = None
stream_active = False

@app.on_event("startup")
async def startup_event():
    global camera, renderer
    try:
        camera = Camera(gpu=True)
        renderer = MediaPipeRenderer()
        print(f"Server started. Static files served from: {STATIC_DIR}")
    except Exception as e:
        print(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    global stream_active
    if camera:
        stream_active = False
        await camera.release_async()

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "viewer.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="viewer.html not found")
    with open(html_path) as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        await streamer.handle_connection(websocket)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

@app.post("/api/detect", response_model=List[DetectionResult])
async def detect_shapes():
    """Process current frame and return detected shapes with 3D dimensions"""
    frame = streamer.get_current_frame()
    if frame is None:
        return []
    
    # Detect objects
    detections = detector.detect(frame)
    
    # Analyze shapes and estimate 3D dimensions
    results = []
    for det in detections:
        shape_info = analyzer.analyze(frame, det.bbox)
        if shape_info:
            results.append(DetectionResult(
                label=det.label,
                confidence=det.confidence,
                shape_3d=shape_info
            ))
    
    return results

@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "camera": "active" if camera and camera._running else "inactive",
        "stream": "active" if stream_active else "inactive",
        "gpu": camera.gpu if camera else False,
        "detection": "ready"
    }

@app.post("/api/control/start")
async def start_stream():
    global stream_active
    try:
        if not stream_active:
            await camera.start()
            stream_active = True
        return {"message": "Stream started", "status": "active"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/control/stop")
async def stop_stream():
    global stream_active
    try:
        if stream_active:
            await camera.stop()
            stream_active = False
        return {"message": "Stream stopped", "status": "inactive"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/original_feed")
async def original_feed():
    async def generate():
        while True:
            if not stream_active:
                await asyncio.sleep(0.1)
                continue
                
            frame = await camera.get_frame_async()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Original feed error: {str(e)}")
            
            await asyncio.sleep(1/30)
            
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/processed_feed")
async def processed_feed():
    async def generate():
        while True:
            if not stream_active:
                await asyncio.sleep(0.1)
                continue
                
            frame = await camera.get_frame_async()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            try:
                # Process with both shape detection and mediapipe rendering
                detections = detector.detect(frame)
                processed_frame = frame.copy()
                
                for det in detections:
                    shape_info = analyzer.analyze(frame, det.bbox)
                    if shape_info:
                        x1, y1, x2, y2 = det.bbox
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        info_text = f"{det.label}: {shape_info.shape_type} ({shape_info.width:.1f}x{shape_info.height:.1f}x{shape_info.depth:.1f})"
                        cv2.putText(processed_frame, info_text, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Add MediaPipe rendering
                _, mp_processed = await run_in_threadpool(renderer.process_frame, processed_frame)
                if mp_processed is not None:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + mp_processed.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Processed feed error: {str(e)}")
            
            await asyncio.sleep(1/30)
            
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)