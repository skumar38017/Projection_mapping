# app/main.py
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from app.webrtc_signaling import signaling_endpoint
from app.config import settings
from app.object_detector import ObjectDetector
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Real-time Object Verification System with Multiple Camera Support"
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
app.mount("/assets", StaticFiles(directory=str(settings.ASSETS_DIR)), name="assets")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page"""
    index_path = settings.STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return HTMLResponse("""
        <html>
            <head><title>Object Verification System</title></head>
            <body>
                <h1>Object Verification System</h1>
                <p>Frontend not found. Please check static/index.html</p>
            </body>
        </html>
        """)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await signaling_endpoint(websocket)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "assets_count": len([f for f in os.listdir(settings.ASSETS_DIR) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    }

@app.get("/api/reference-images")
async def get_reference_images():
    """Get list of reference images"""
    images = []
    
    for filename in os.listdir(settings.ASSETS_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = settings.ASSETS_DIR / filename
            file_size = os.path.getsize(filepath)
            
            images.append({
                "filename": filename,
                "size": file_size,
                "url": f"/assets/{filename}"
            })
    
    return {"images": images}

@app.get("/api/settings")
async def get_settings():
    """Get current detection settings"""
    return {
        "feature_threshold": settings.FEATURE_MATCH_THRESHOLD,
        "deep_threshold": settings.DEEP_MATCH_THRESHOLD,
        "detection_confidence": settings.DETECTION_CONFIDENCE,
        "camera_width": settings.DEFAULT_CAMERA_WIDTH,
        "camera_height": settings.DEFAULT_CAMERA_HEIGHT,
        "fps": settings.DEFAULT_FPS
    }

@app.post("/api/test-detector")
async def test_detector():
    """Test if the object detector can be initialized"""
    try:
        detector = ObjectDetector()
        ref_images = detector.get_reference_images_info()
        
        return {
            "status": "success",
            "reference_images_loaded": len(ref_images),
            "feature_extractor": detector.feature_extractor is not None,
            "deep_extractor": detector.deep_feature_extractor is not None,
            "detection_model": detector.detection_model is not None
        }
    except Exception as e:
        logger.error(f"Detector test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug")
async def debug_info():
    """Debug endpoint to check system status"""
    try:
        from app.webrtc_signaling import camera_manager, get_reference_images_list
        
        # Get camera info
        cameras = camera_manager.get_available_cameras()
        
        # Get reference images
        ref_images = get_reference_images_list()
        
        return {
            "status": "ok",
            "cameras_found": len(cameras),
            "cameras": cameras,
            "reference_images_found": len(ref_images),
            "reference_images": [{"filename": img["filename"], "size": img["size"]} for img in ref_images],
            "assets_dir": str(settings.ASSETS_DIR),
            "assets_exists": settings.ASSETS_DIR.exists(),
            "static_dir": str(settings.STATIC_DIR),
            "static_exists": settings.STATIC_DIR.exists()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "assets_dir": str(settings.ASSETS_DIR),
            "static_dir": str(settings.STATIC_DIR)
        }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Assets directory: {settings.ASSETS_DIR}")
    logger.info(f"Static directory: {settings.STATIC_DIR}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
