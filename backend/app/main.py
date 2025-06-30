# ~/app/main.py
import logging
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pathlib import Path
import asyncio
import uvloop
from concurrent.futures import ThreadPoolExecutor

# Import components
from app.camera_factory import CameraFactory
from app.camera.stream import WebRTCStreamer
from app.detection.shape_analyzer import ShapeAnalyzer
from app.detection.object_detector import ObjectDetector
from app.schemas.detection import DetectionResult
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

uvloop.install()
app = FastAPI(
    title="3D Shape Detection API",
    description="Advanced 3D shape detection with real-time streaming",
    version="1.0.0"
)

# Configuration
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

if not STATIC_DIR.exists():
    raise RuntimeError(f"Static directory not found at {STATIC_DIR}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
executor = ThreadPoolExecutor(max_workers=4)

# Initialize components
camera = None
detector = ObjectDetector()
analyzer = ShapeAnalyzer()
streamer = WebRTCStreamer(detector, analyzer)
stream_active = False

@app.on_event("startup")
async def startup():
    global camera
    try:
        camera = CameraFactory.create_camera()
        if camera is None:
            raise RuntimeError("Failed to initialize any camera")
        
        logger.info(f"Using camera: {camera.get_camera_info()}")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown():
    global stream_active
    stream_active = False
    if camera:
        camera.release()
    logger.info("Application shutdown")

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open(STATIC_DIR / "viewer.html") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Failed to serve viewer.html: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load viewer")

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        await streamer.handle_connection(websocket)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

@app.post("/api/detect", response_model=List[DetectionResult])
async def detect_objects():
    if not camera:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    try:
        frame = camera.get_frame()
        depth = camera.get_depth_frame()
        
        if frame is None:
            raise HTTPException(status_code=503, detail="No frame available")
        
        detections = detector.detect(frame)
        results = []
        
        for det in detections:
            shape = analyzer.analyze(frame, depth, det)
            if shape:
                results.append(DetectionResult(
                    label=det.label,
                    confidence=det.confidence,
                    shape_3d=shape
                ))
        
        return results
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "camera": camera.get_camera_info() if camera else "none",
        "stream": "active" if stream_active else "inactive",
        "depth_capable": camera.has_depth_capability() if camera else False
    }

@app.post("/api/control/start")
async def start_stream():
    global stream_active
    try:
        if not stream_active and camera:
            camera.start()
            stream_active = True
            logger.info("Stream started")
        return {"message": "Stream started", "status": "active"}
    except Exception as e:
        logger.error(f"Failed to start stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/control/stop")
async def stop_stream():
    global stream_active
    try:
        if stream_active and camera:
            camera.stop()
            stream_active = False
            logger.info("Stream stopped")
        return {"message": "Stream stopped", "status": "inactive"}
    except Exception as e:
        logger.error(f"Failed to stop stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/original_feed")
async def original_feed():
    async def generate():
        while True:
            if not stream_active or not camera:
                await asyncio.sleep(0.1)
                continue
                
            frame = camera.get_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                logger.error(f"Frame encoding error: {str(e)}")
                break
            
            await asyncio.sleep(1/30)
            
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/processed_feed")
async def processed_feed():
    async def generate():
        while True:
            if not stream_active or not camera:
                await asyncio.sleep(0.1)
                continue
                
            frame = camera.get_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            try:
                # Process the frame with your detector and analyzer
                detections = detector.detect(frame)
                processed_frame = frame.copy()
                
                for det in detections:
                    shape = analyzer.analyze(frame, None, det)  # Passing None for depth since you're using webcam
                    if shape:
                        # Draw bounding box
                        x1, y1, x2, y2 = det.bbox
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{det.label} ({det.confidence:.2f})"
                        cv2.putText(processed_frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                _, buffer = cv2.imencode('.jpg', processed_frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                logger.error(f"Frame processing error: {str(e)}")
                break
            
            await asyncio.sleep(1/30)
            
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)