# ~/app/routes/router.py
from fastapi import APIRouter, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pathlib import Path
import asyncio
import cv2
import time
import datetime
import threading
from app.camera import AsyncCameraCapture as CameraCapture
from typing import Dict, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor 
from app.camera_factory import CameraFactory
from app.camera.stream import WebRTCStreamer
from app.detection.shape_analyzer import ShapeAnalyzer
from app.detection.object_detector import ObjectDetector
from app.schemas.detection import DetectionResult
from app.network.osc_output import OSCOutput
from app.network.lan_output import LANOutput

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components (these will be injected from main.py)
camera = None
detector = ObjectDetector()
analyzer = ShapeAnalyzer()
streamer = WebRTCStreamer(detector, analyzer)
stream_active = False
lan_output = None
osc_output = None
executor = None
STATIC_DIR = None

def init_routers(
    cam: CameraFactory,
    lan: LANOutput,
    osc: OSCOutput,
    exec: ThreadPoolExecutor,
    static_dir: Path
):
    global camera, lan_output, osc_output, executor, STATIC_DIR
    camera = cam
    lan_output = lan
    osc_output = osc
    executor = exec
    STATIC_DIR = static_dir

@router.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open(STATIC_DIR / "viewer.html") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Failed to serve viewer.html: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load viewer")

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        await streamer.handle_connection(websocket)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

@router.post("/api/detect", response_model=List[DetectionResult])
async def detect_objects():
    if not camera:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    try:
        frame = camera.get_frame()
        depth = camera.get_depth_frame()
        
        if frame is None:
            raise HTTPException(status_code=503, detail="No frame available")
        
        # Get detection and timing data
        start_time = time.time()
        detections = detector.detect(frame)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        results = []
        
        for det in detections:
            shape = analyzer.analyze(frame, depth, det)
            if shape:
                result = DetectionResult(
                    label=det.label,
                    confidence=det.confidence,
                    shape_3d=shape
                )
                results.append(result)
                
                # Prepare the performance data
                perf_data = {
                    'resolution': f"{frame.shape[0]}x{frame.shape[1]}",
                    'inference_time': inference_time,
                    'num_detections': len(detections),
                    'detection': result.dict(),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Send detection over LAN
                try:
                    lan_output.async_send({
                        'type': 'detection',
                        'data': perf_data
                    })
                except Exception as e:
                    logger.error(f"Failed to send LAN data: {str(e)}")
                
                # Send detection via OSC
                try:
                    osc_data = {
                        'label': det.label,
                        'confidence': float(det.confidence),
                        'x': (det.bbox[0] + det.bbox[2]) / 2 / frame.shape[1],
                        'y': (det.bbox[1] + det.bbox[3]) / 2 / frame.shape[0],
                        'width': (det.bbox[2] - det.bbox[0]) / frame.shape[1],
                        'height': (det.bbox[3] - det.bbox[1]) / frame.shape[0],
                        'shape_type': shape.shape_type,
                        'shape_width': float(shape.width),
                        'shape_height': float(shape.height),
                        'shape_depth': float(shape.depth),
                        'inference_time': inference_time,
                        'resolution': f"{frame.shape[0]}x{frame.shape[1]}"
                    }
                    osc_output.async_send("/detection", osc_data)
                except Exception as e:
                    logger.error(f"Failed to send OSC data: {str(e)}")

        # Log the performance data
        logger.info(
            f"0: {frame.shape[0]}x{frame.shape[1]} "
            f"{len(detections)} objects, {inference_time:.1f}ms"
        )

        return results

    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/osc_output")
async def send_osc_output(address: str = "/data", data: Dict[str, Any] = None):
    """Manual endpoint to send custom data via OSC"""
    if data is None:
        data = {}
    osc_output.async_send(address, data)
    return {"message": f"Data sent via OSC to {address}"}

@router.post("/api/lan_output")
async def send_lan_output(data: Dict[str, Any]):
    """Manual endpoint to send custom data over LAN"""
    lan_output.async_send(data)
    return {"message": "Data sent to LAN"}

@router.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "camera": camera.get_camera_info() if camera else "none",
        "stream": "active" if stream_active and camera and camera.is_capturing() else "inactive",
        "depth_capable": camera.has_depth_capability() if camera else False,
        "lan_connected": lan_output.socket is not None if hasattr(lan_output, 'socket') else False,
        "osc_connected": osc_output.client is not None,
        "lan_debug": lan_output.debug,
        "osc_debug": osc_output.debug
    }

@router.get("/api/threads")
async def get_thread_status():
    threads = []
    for thread in threading.enumerate():
        threads.append({
            "name": thread.name,
            "alive": thread.is_alive(),
            "daemon": thread.daemon
        })
    return threads

@router.post("/api/control/start")
async def start_stream():
    global stream_active
    try:
        if not stream_active and camera:
            await camera.start()  # Add await
            stream_active = True
            logger.info("Stream started")
        return {"message": "Stream started", "status": "active"}
    except Exception as e:
        logger.error(f"Failed to start stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/control/stop")
async def stop_stream():
    global stream_active
    try:
        if stream_active and camera:
            await camera.stop()  # Add await
            stream_active = False
            logger.info("Stream stopped")
        return {"message": "Stream stopped", "status": "inactive"}
    except Exception as e:
        logger.error(f"Failed to stop stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/original_feed")
async def original_feed():
    async def generate():
        while True:
            if not stream_active or not camera:
                await asyncio.sleep(0.1)
                continue
                
            frame = await camera.get_frame()
            if frame is None:
                # Return a black frame if no valid frame available
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                logger.error(f"Frame encoding error: {str(e)}")
                break
            
            await asyncio.sleep(1/30)  # Control frame rate
            
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/processed_feed")
async def processed_feed():
    async def generate():
        while True:
            if not stream_active or not camera:
                await asyncio.sleep(0.1)
                continue
                
            frame = await camera.get_frame()
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            try:
                # Process frame only if it's not the default black frame
                if frame.any():  # Checks if frame has any non-zero values
                    detections = detector.detect(frame)
                    for det in detections:
                        # Your detection drawing code here
                        pass
                
                _, buffer = cv2.imencode('.jpg', frame)
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

@router.get("/api/test/lan")
async def test_lan_output():
    """Test endpoint for LAN output"""
    test_data = {
        "test": "LAN_TEST",
        "timestamp": datetime.datetime.now().isoformat(),
        "values": [1.2, 3.4, 5.6]
    }
    lan_output.async_send(test_data)
    return {"message": "LAN test data sent", "data": test_data}

@router.get("/api/test/osc")
async def test_osc_output():
    """Test endpoint for OSC output"""
    test_data = {
        "test": "OSC_TEST",
        "timestamp": datetime.datetime.now().isoformat(),
        "values": [7.8, 9.0, 1.2]
    }
    osc_output.async_send("/test", test_data)
    return {"message": "OSC test data sent", "data": test_data}