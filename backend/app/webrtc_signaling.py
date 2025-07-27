# app/webrtc_signaling.py
import base64
import cv2
import numpy as np
import json
import asyncio
import time
from fastapi import WebSocket, WebSocketDisconnect
from app.simple_matcher import simple_object_matching, get_simple_matcher
from app.config import settings
from app.utils import image_to_base64, make_json_serializable
from app.resource_monitor import resource_monitor
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        self.available_cameras = []
        self.active_camera = None
        self.is_streaming = False
        self.detector = None
        self._scan_cameras()
    
    def _scan_cameras(self):
        """Scan for available cameras"""
        self.available_cameras = []
        
        # Check for cameras (0-10 range)
        for i in range(10):
            cap = None
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Try to read a frame to verify camera works
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        
                        # Set default values if properties are 0
                        if width == 0:
                            width = 640
                        if height == 0:
                            height = 480
                        if fps == 0:
                            fps = 30
                        
                        camera_info = {
                            'id': i,
                            'name': f'Camera {i}',
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'available': True
                        }
                        self.available_cameras.append(camera_info)
                        logger.info(f"Found camera {i}: {width}x{height} @ {fps}fps")
            except Exception as e:
                logger.debug(f"Camera {i} not available: {e}")
            finally:
                if cap is not None:
                    cap.release()
        
        # Add default camera if no cameras found but system has video devices
        if not self.available_cameras:
            # Try to add a default camera entry
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    camera_info = {
                        'id': 0,
                        'name': 'Default Camera',
                        'width': 640,
                        'height': 480,
                        'fps': 30,
                        'available': True
                    }
                    self.available_cameras.append(camera_info)
                    logger.info("Added default camera")
                cap.release()
            except Exception as e:
                logger.error(f"Failed to add default camera: {e}")
        
        if not self.available_cameras:
            logger.warning("No cameras found")
        else:
            logger.info(f"Found {len(self.available_cameras)} cameras")
    
    def get_available_cameras(self):
        """Get list of available cameras"""
        return self.available_cameras
    
    def start_camera(self, camera_id=0):
        """Start camera streaming"""
        try:
            print(f"\n📷 [CAMERA] Starting camera {camera_id}...")
            
            if self.active_camera is not None:
                print(f"🔄 [CAMERA] Stopping existing camera...")
                self.stop_camera()
            
            self.active_camera = cv2.VideoCapture(camera_id)
            if not self.active_camera.isOpened():
                raise Exception(f"Cannot open camera {camera_id}")
            
            # Set camera properties
            self.active_camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.DEFAULT_CAMERA_WIDTH)
            self.active_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.DEFAULT_CAMERA_HEIGHT)
            self.active_camera.set(cv2.CAP_PROP_FPS, settings.DEFAULT_FPS)
            
            # Get actual properties
            actual_width = int(self.active_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.active_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.active_camera.get(cv2.CAP_PROP_FPS))
            
            self.is_streaming = True
            # Use simple matcher - exactly what you requested
            self.matcher = get_simple_matcher()
            
            print(f"✅ [CAMERA] Camera {camera_id} started successfully")
            print(f"   📐 Resolution: {actual_width}x{actual_height}")
            print(f"   🎬 FPS: {actual_fps}")
            print(f"   🔍 Matcher: Simple object matching ready")
            
            logger.info(f"Started camera {camera_id}")
            return True
            
        except Exception as e:
            print(f"💥 [CAMERA] Failed to start camera {camera_id}: {e}")
            logger.error(f"Failed to start camera {camera_id}: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera streaming"""
        try:
            print(f"\n🛑 [CAMERA] Stopping camera...")
            self.is_streaming = False
            if self.active_camera is not None:
                self.active_camera.release()
                self.active_camera = None
            print(f"✅ [CAMERA] Camera stopped successfully")
            logger.info("Camera stopped")
            return True
        except Exception as e:
            print(f"💥 [CAMERA] Failed to stop camera: {e}")
            logger.error(f"Failed to stop camera: {e}")
            return False
    
    def get_frame(self):
        """Get current frame from active camera"""
        if not self.is_streaming or self.active_camera is None:
            return None
        
        try:
            ret, frame = self.active_camera.read()
            if ret:
                return frame
            else:
                logger.warning("Failed to read frame from camera")
                return None
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None

# Global camera manager
camera_manager = CameraManager()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.streaming_task = None
        self.match_history = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"\n🔗 [CONNECTION] Client connected from {websocket.client}")
        print(f"👥 [CLIENTS] Total connections: {len(self.active_connections)}")
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"\n❌ [DISCONNECT] Client disconnected")
        print(f"👥 [CLIENTS] Total connections: {len(self.active_connections)}")
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            # Make message JSON serializable
            serializable_message = make_json_serializable(message)
            await websocket.send_text(json.dumps(serializable_message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                # Make message JSON serializable
                serializable_message = make_json_serializable(message)
                await connection.send_text(json.dumps(serializable_message))
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def start_streaming(self):
        """Start continuous camera streaming"""
        if self.streaming_task is not None:
            return
        
        self.streaming_task = asyncio.create_task(self._streaming_loop())
        logger.info("Started streaming task")
    
    async def stop_streaming(self):
        """Stop continuous camera streaming"""
        if self.streaming_task is not None:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
            self.streaming_task = None
        logger.info("Stopped streaming task")
    
    async def _streaming_loop(self):
        """Main streaming loop with detailed logging"""
        print(f"\n🎬 [STREAMING] Started streaming loop at {datetime.now().strftime('%H:%M:%S')}")
        frame_count = 0
        
        try:
            while camera_manager.is_streaming:
                frame_count += 1
                print(f"\n📹 [FRAME {frame_count}] Capturing frame...")
                
                frame = camera_manager.get_frame()
                if frame is not None:
                    print(f"✅ [CAMERA] Got frame: {frame.shape}")
                    
                    try:
                        # Record frame processing start time
                        frame_start_time = time.time()
                        
                        # Simple matching: stored images vs real-time camera object
                        result, processed_frame = simple_object_matching(frame)
                        
                        # Record frame processing time for resource monitoring
                        frame_processing_time = time.time() - frame_start_time
                        resource_monitor.record_frame_time(frame_processing_time)
                        
                        # Convert frames to base64
                        print(f"🔄 [ENCODING] Converting frames to base64...")
                        original_b64 = image_to_base64(frame)
                        processed_b64 = image_to_base64(processed_frame)
                        print(f"✅ [ENCODING] Frames encoded successfully")
                        
                        # Store match history
                        if result["success"] and result.get("match_found", False):
                            print(f"📝 [HISTORY] Storing match record...")
                            match_record = {
                                "timestamp": result["timestamp"],
                                "match": result["matched_image"],
                                "score": float(result["similarity_score"]),
                                "method": result.get("method", "unknown")
                            }
                            self.match_history.append(match_record)
                            
                            # Keep only last 50 matches
                            if len(self.match_history) > 50:
                                self.match_history = self.match_history[-50:]
                            
                            print(f"✅ [HISTORY] Match stored. Total history: {len(self.match_history)}")
                        
                        # Broadcast to all connected clients
                        message = {
                            "type": "stream_frame",
                            "original_frame": original_b64,
                            "processed_frame": processed_b64,
                            "result": result,
                            "timestamp": datetime.now().isoformat(),
                            "performance": {
                                "processing_time": frame_processing_time,
                                "fps": resource_monitor.average_fps,
                                "mode": resource_monitor.performance_mode,
                                "frame_number": frame_count
                            }
                        }
                        
                        print(f"🔍 [DEBUG] Result object being sent: {result}")
                        print(f"🔍 [DEBUG] Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                        print(f"📡 [BROADCAST] Sending to {len(self.active_connections)} clients...")
                        await self.broadcast(message)
                        print(f"✅ [BROADCAST] Frame {frame_count} sent successfully")
                        
                    except Exception as e:
                        print(f"💥 [ERROR] Frame processing failed: {e}")
                        logger.error(f"Error processing frame: {e}")
                        # Continue streaming even if one frame fails
                        continue
                
                else:
                    print(f"⚠️ [CAMERA] No frame received")
                
                # Control frame rate dynamically based on performance
                optimal_fps = resource_monitor.get_optimal_fps()
                sleep_time = 1/optimal_fps
                print(f"⏱️ [FPS] Sleeping {sleep_time*1000:.1f}ms for {optimal_fps} FPS")
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            print(f"\n🛑 [STREAMING] Loop cancelled after {frame_count} frames")
            logger.info("Streaming loop cancelled")
        except Exception as e:
            print(f"\n💥 [STREAMING] Fatal error after {frame_count} frames: {e}")
            logger.error(f"Error in streaming loop: {e}")
            # Try to notify clients about the error
            try:
                error_message = {
                    "type": "error",
                    "message": f"Streaming error: {str(e)}"
                }
                await self.broadcast(error_message)
            except:
                pass  # Don't let error handling cause more errors

# Global connection manager
manager = ConnectionManager()

async def signaling_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # Send initial data
        await manager.send_personal_message({
            "type": "init",
            "cameras": camera_manager.get_available_cameras(),
            "reference_images": get_reference_images_list(),
            "settings": {
                "feature_threshold": settings.FEATURE_MATCH_THRESHOLD,
                "deep_threshold": settings.DEEP_MATCH_THRESHOLD,
                "detection_confidence": settings.DETECTION_CONFIDENCE
            }
        }, websocket)
        
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_message(message, websocket)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, websocket)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": str(e)
                }, websocket)
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

async def handle_message(message: dict, websocket: WebSocket):
    """Handle different types of messages from client"""
    msg_type = message.get("type")
    
    if msg_type == "start_camera":
        camera_id = message.get("camera_id", 0)
        success = camera_manager.start_camera(camera_id)
        
        if success:
            await manager.start_streaming()
            await manager.send_personal_message({
                "type": "camera_started",
                "camera_id": camera_id,
                "success": True
            }, websocket)
        else:
            await manager.send_personal_message({
                "type": "camera_started",
                "camera_id": camera_id,
                "success": False,
                "error": f"Failed to start camera {camera_id}"
            }, websocket)
    
    elif msg_type == "stop_camera":
        await manager.stop_streaming()
        camera_manager.stop_camera()
        
        await manager.send_personal_message({
            "type": "camera_stopped",
            "success": True
        }, websocket)
    
    elif msg_type == "get_cameras":
        camera_manager._scan_cameras()  # Rescan cameras
        await manager.send_personal_message({
            "type": "cameras_list",
            "cameras": camera_manager.get_available_cameras()
        }, websocket)
    
    elif msg_type == "get_reference_images":
        await manager.send_personal_message({
            "type": "reference_images",
            "images": get_reference_images_list()
        }, websocket)
    
    elif msg_type == "get_match_history":
        await manager.send_personal_message({
            "type": "match_history",
            "history": manager.match_history
        }, websocket)
    
    elif msg_type == "single_frame":
        # Process single frame from client
        try:
            img_data = base64.b64decode(message["frame"])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                result, processed_frame = verify_object_from_frame(frame)
                processed_b64 = image_to_base64(processed_frame)
                
                await manager.send_personal_message({
                    "type": "single_frame_result",
                    "result": result,
                    "processed_frame": processed_b64
                }, websocket)
            else:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Failed to decode frame"
                }, websocket)
                
        except Exception as e:
            await manager.send_personal_message({
                "type": "error",
                "message": f"Error processing frame: {str(e)}"
            }, websocket)
    
    else:
        await manager.send_personal_message({
            "type": "error",
            "message": f"Unknown message type: {msg_type}"
        }, websocket)

def get_reference_images_list():
    """Get list of reference images with metadata"""
    images = []
    
    try:
        # Check if assets directory exists
        if not settings.ASSETS_DIR.exists():
            logger.warning(f"Assets directory does not exist: {settings.ASSETS_DIR}")
            return images
        
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        image_files = [f for f in os.listdir(settings.ASSETS_DIR) 
                      if f.lower().endswith(image_extensions)]
        
        logger.info(f"Found {len(image_files)} image files in assets directory")
        
        for filename in image_files:
            filepath = settings.ASSETS_DIR / filename
            
            try:
                # Get image info
                img = cv2.imread(str(filepath))
                if img is not None:
                    height, width = img.shape[:2]
                    file_size = os.path.getsize(filepath)
                    
                    # Convert to base64 for display
                    img_b64 = image_to_base64(img)
                    
                    images.append({
                        "filename": filename,
                        "width": width,
                        "height": height,
                        "size": file_size,
                        "image_data": img_b64
                    })
                    logger.debug(f"Processed reference image: {filename} ({width}x{height})")
                else:
                    logger.warning(f"Could not read image: {filename}")
            except Exception as e:
                logger.error(f"Error processing reference image {filename}: {e}")
    
    except Exception as e:
        logger.error(f"Error scanning assets directory: {e}")
    
    logger.info(f"Successfully loaded {len(images)} reference images")
    return images