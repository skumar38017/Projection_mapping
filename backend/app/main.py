# ~/app/main.py
import logging
from typing import Dict, Any, List, Union
import time
import datetime
import threading
import cv2
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles  # Add this import
from pathlib import Path
import asyncio
import uvloop
from concurrent.futures import ThreadPoolExecutor
from app.network.lan_output import LANOutput
from app.network.osc_output import OSCOutput
from app.camera_factory import CameraFactory
from app.routes.router import router, init_routers

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
        
        # Initialize routers with dependencies
        init_routers(camera, lan_output, osc_output, executor, STATIC_DIR)
        
        # Include the router
        app.include_router(router)
        
        # Start continuous outputs
        start_continuous_outputs()
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown():
    if camera:
        camera.release()
    lan_output.close()
    osc_output.close()
    logger.info("Application shutdown")
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)