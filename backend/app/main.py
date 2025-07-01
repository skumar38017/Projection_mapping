# ~/app/main.py
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import logging
import asyncio
import uvloop
from concurrent.futures import ThreadPoolExecutor
import time
import datetime
import numpy as np
from app.camera_factory import CameraFactory
from app.network.lan_output import LANOutput
from app.network.osc_output import OSCOutput
from app.routes.router import router, init_routers

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize logging and uvloop
logging.basicConfig(level=logging.INFO)
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
lan_output = None
osc_output = None
stream_active = False

async def camera_processing_loop():
    """Main processing loop that handles camera frames and data output"""
    global camera, lan_output, osc_output, stream_active
    
    while stream_active:
        try:
            if not camera:
                await asyncio.sleep(0.1)
                continue
                
            frame = await camera.get_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
                
            # Process the frame (detections, etc)
            # This is where you'd add your processing logic
            
            # Create data packet
            frame_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                "frame_size": frame.size,
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1
            }
            
            # Print to terminal
            logger.info(f"Frame data: {frame_data}")
            
            # Send via LAN
            if lan_output:
                try:
                    lan_output.async_send({
                        "type": "frame_data",
                        "data": frame_data
                    })
                except Exception as e:
                    logger.error(f"LAN send error: {str(e)}")
            
            # Send via OSC
            if osc_output:
                try:
                    osc_output.async_send("/frame", frame_data)
                except Exception as e:
                    logger.error(f"OSC send error: {str(e)}")
                    
            await asyncio.sleep(1/30)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"Processing loop error: {str(e)}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup():
    global camera, lan_output, osc_output, stream_active
    
    try:
        # Initialize network outputs first
        lan_output = LANOutput(host='255.255.255.255', port=5000, debug=True)
        osc_output = OSCOutput(ip='localhost', port=5005, debug=True)
        
        # Initialize camera
        camera = await CameraFactory.create_camera()
        if not camera:
            raise RuntimeError("Failed to initialize camera")
        
        # Initialize routers
        init_routers(camera, lan_output, osc_output, executor, STATIC_DIR)
        app.include_router(router)
        
        # Start processing loop
        stream_active = True
        asyncio.create_task(camera_processing_loop())
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown():
    global stream_active
    
    stream_active = False
    if camera:
        await camera.stop()
    
    if lan_output:
        lan_output.close()
    
    if osc_output:
        osc_output.close()
    
    logger.info("Application shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)