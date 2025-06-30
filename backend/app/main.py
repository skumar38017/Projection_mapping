# ~/app/main.py
import logging
from typing import Dict, Any, List, Union
import time
import datetime
import threading
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pathlib import Path
import asyncio
import uvloop
from concurrent.futures import ThreadPoolExecutor
from app.network.lan_output import LANOutput

# Import components
from app.camera_factory import CameraFactory
from app.camera.stream import WebRTCStreamer
from app.detection.shape_analyzer import ShapeAnalyzer
from app.detection.object_detector import ObjectDetector
from app.schemas.detection import DetectionResult
from typing import List
from app.network.osc_output import OSCOutput

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

# Enable debug printing for network outputs
lan_output = LANOutput(host='255.255.255.255', port=5000, debug=True)
osc_output = OSCOutput(ip='localhost', port=5005, debug=True)

def start_continuous_outputs():
    """Start continuous LAN and OSC output in background threads with error handling"""
    def lan_continuous():
        while True:
            try:
                test_data = {
                    "type": "continuous_lan",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "values": [1.2, 3.4, 5.6]
                }
                lan_output.send_data(test_data)
                logger.debug("Sent continuous LAN data")
            except Exception as e:
                logger.error(f"LAN continuous output error: {str(e)}")
            time.sleep(1.0)
            
    def osc_continuous():
        while True:
            try:
                test_data = {
                    "test": "continuous_osc",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "values": [7.8, 9.0, 1.2]
                }
                osc_output.send_data("/continuous", test_data)
                logger.debug("Sent continuous OSC data")
            except Exception as e:
                logger.error(f"OSC continuous output error: {str(e)}")
            time.sleep(1.0)
    
    # Start both continuous outputs
    executor.submit(lan_continuous)
    executor.submit(osc_continuous)
    logger.info("Started continuous LAN and OSC output threads")

@app.on_event("startup")
async def startup():
    global camera
    try:
        camera = CameraFactory.create_camera()
        if camera is None:
            raise RuntimeError("Failed to initialize any camera")
        
        logger.info(f"Using camera: {camera.get_camera_info()}")
        # Start continuous outputs
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown():
    global stream_active
    stream_active = False   
    if camera:
        camera.release()
    lan_output.close()
    osc_output.close()
    logger.info("Application shutdown")
    executor.shutdown(wait=True)

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

        # Log the performance data (this matches your example output)
        logger.info(
            f"0: {frame.shape[0]}x{frame.shape[1]} "
            f"{len(detections)} objects, {inference_time:.1f}ms\n"
            f"Speed: {preprocess_time:.1f}ms preprocess, "
            f"{inference_time:.1f}ms inference, "
            f"{postprocess_time:.1f}ms postprocess per image at shape (1, 3, {frame.shape[0]}, {frame.shape[1]})"
        )

        return results

    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/osc_output")
async def send_osc_output(address: str = "/data", data: Dict[str, Any] = None):
    """Manual endpoint to send custom data via OSC"""
    if data is None:
        data = {}
    osc_output.async_send(address, data)
    return {"message": f"Data sent via OSC to {address}"}

@app.post("/api/lan_output")
async def send_lan_output(data: Dict[str, Any]):
    """Manual endpoint to send custom data over LAN"""
    lan_output.async_send(data)
    return {"message": "Data sent to LAN"}

@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "camera": camera.get_camera_info() if camera else "none",
        "stream": "active" if stream_active else "inactive",
        "depth_capable": camera.has_depth_capability() if camera else False,
        "lan_connected": lan_output.socket is not None if hasattr(lan_output, 'socket') else False,
        "osc_connected": osc_output.client is not None,
        "lan_debug": lan_output.debug,
        "osc_debug": osc_output.debug
    }

@app.get("/api/threads")
async def get_thread_status():
    threads = []
    for thread in threading.enumerate():
        threads.append({
            "name": thread.name,
            "alive": thread.is_alive(),
            "daemon": thread.daemon
        })
    return threads

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
                start_time = time.time()
                detections = detector.detect(frame)
                inference_time = (time.time() - start_time) * 1000
                
                processed_frame = frame.copy()
                
                for det in detections:
                    shape = analyzer.analyze(frame, None, det)
                    if shape:
                        # Draw bounding box
                        x1, y1, x2, y2 = det.bbox
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{det.label} ({det.confidence:.2f})"
                        cv2.putText(processed_frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Send detection data
                        perf_data = {
                            'resolution': f"{frame.shape[0]}x{frame.shape[1]}",
                            'inference_time': inference_time,
                            'detection': {
                                'label': det.label,
                                'confidence': det.confidence,
                                'bbox': det.bbox,
                                'shape': shape.shape_type if shape else None
                            }
                        }
                        
                        lan_output.async_send(perf_data)
                        osc_output.async_send("/detection", {
                            **perf_data['detection'],
                            'inference_time': perf_data['inference_time'],
                            'resolution': perf_data['resolution']
                        })
                
                # Log performance
                logger.info(
                    f"0: {frame.shape[0]}x{frame.shape[1]} "
                    f"{len(detections)} objects, {inference_time:.1f}ms"
                )
                
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

@app.get("/api/test/lan")
async def test_lan_output():
    """Test endpoint for LAN output"""
    test_data = {
        "test": "LAN_TEST",
        "timestamp": datetime.datetime.now().isoformat(),
        "values": [1.2, 3.4, 5.6]
    }
    lan_output.async_send(test_data)
    return {"message": "LAN test data sent", "data": test_data}

@app.get("/api/test/osc")
async def test_osc_output():
    """Test endpoint for OSC output"""
    test_data = {
        "test": "OSC_TEST",
        "timestamp": datetime.datetime.now().isoformat(),
        "values": [7.8, 9.0, 1.2]
    }
    osc_output.async_send("/test", test_data)
    return {"message": "OSC test data sent", "data": test_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)