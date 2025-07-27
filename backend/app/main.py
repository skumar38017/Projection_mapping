#!/usr/bin/env python3
"""
Unified Projection Mapping Server
FastAPI application with startup functionality merged
Run with: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
Or: python app/main.py (for direct startup)
"""

import os
import sys
import logging

# Add parent directory to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FastAPI imports
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# App imports
from app.webrtc_signaling import signaling_endpoint
from app.config import settings
from app.object_detector import ObjectDetector
from app.resource_monitor import resource_monitor, start_resource_monitoring
from app.network_broadcaster import get_network_broadcaster
from pathlib import Path

# Startup functionality imports
import subprocess
import argparse
import socket
import psutil

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
    description="Real-time Object Verification System with Network Broadcasting and Dynamic Resources"
)

# Start resource monitoring
start_resource_monitoring()

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

@app.get("/api/network")
async def get_network_info():
    """Get network broadcasting information"""
    broadcaster = get_network_broadcaster()
    return broadcaster.get_network_info()

@app.get("/api/settings")
async def get_settings():
    """Get current detection settings with dynamic recommendations"""
    dynamic_config = resource_monitor.get_resource_status()
    network_info = get_network_broadcaster().get_network_info()
    
    return {
        "feature_threshold": settings.FEATURE_MATCH_THRESHOLD,
        "deep_threshold": settings.DEEP_MATCH_THRESHOLD,
        "detection_confidence": settings.DETECTION_CONFIDENCE,
        "camera_width": settings.DEFAULT_CAMERA_WIDTH,
        "camera_height": settings.DEFAULT_CAMERA_HEIGHT,
        "fps": settings.DEFAULT_FPS,
        "dynamic_memory": settings.DYNAMIC_MEMORY,
        "performance_mode": dynamic_config["performance"]["mode"],
        "recommended_resolution": dynamic_config["performance"]["optimal_resolution"],
        "recommended_fps": dynamic_config["recommendations"]["optimal_fps"],
        "network": network_info
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

# Startup functionality
def get_local_ip():
    """Get local machine IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

def setup_environment():
    """Setup environment for FULL system resource utilization"""
    local_ip = get_local_ip()
    
    # Common environment
    os.environ['SIMPLE_MATCHING_MODE'] = 'true'
    os.environ['NETWORK_BROADCASTING'] = 'true'
    os.environ['LOCAL_IP'] = local_ip
    os.environ['DETAILED_LOGGING'] = 'true'
    
    # FULL GPU utilization - no limits
    os.environ['USE_GPU'] = 'true'  # Force GPU usage
    os.environ['DYNAMIC_MEMORY'] = 'true'
    os.environ['FORCE_CPU_TENSORFLOW'] = 'false'  # Use GPU for TensorFlow
    os.environ['FORCE_CPU_PYTORCH'] = 'false'    # Use GPU for PyTorch
    
    # Remove any GPU memory limits
    if 'GPU_MEMORY_LIMIT' in os.environ:
        del os.environ['GPU_MEMORY_LIMIT']
    
    # TensorFlow FULL GPU configuration
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Show more info
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable optimizations
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'  # Use all GPU memory
    
    # PyTorch FULL GPU configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:0'  # No memory fragmentation limits
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async GPU operations
    
    # FULL CPU utilization
    cpu_count = psutil.cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Use ALL CPU cores
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
    
    # OpenCV optimizations for full CPU usage
    os.environ['OPENCV_NUM_THREADS'] = str(cpu_count)
    
    # Real-time processing with full resources
    os.environ['REALTIME_MODE'] = 'true'
    os.environ['USE_REALTIME_DETECTOR'] = 'true'
    os.environ['TARGET_FPS'] = '30'
    os.environ['MAX_PROCESSING_TIME'] = '0.020'  # Faster processing target
    
    # System-wide performance optimizations
    os.environ['MALLOC_ARENA_MAX'] = '4'  # Memory allocation optimization
    
    return local_ip, cpu_count, total_memory_gb

def check_system_resources():
    """Check system resources"""
    try:
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        
        return {
            'cpu_count': cpu_count,
            'cpu_usage': cpu_usage,
            'memory_gb': memory_gb,
            'memory_available_gb': memory_available_gb,
            'disk_free_gb': disk_free_gb
        }
    except:
        return None

def check_gpu_status():
    """Check GPU status"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                if line.strip():
                    name, total_mem, free_mem = line.split(', ')
                    gpu_info.append({
                        'id': i,
                        'name': name,
                        'total_mb': int(total_mem),
                        'free_mb': int(free_mem)
                    })
            return gpu_info
        return None
    except:
        return None

def check_dependencies():
    """Check real-time dependencies"""
    deps = {
        'mediapipe': False,
        'ultralytics': False,
        'python_osc': False,
        'netifaces': False
    }
    
    try:
        import mediapipe
        deps['mediapipe'] = True
    except ImportError:
        pass
    
    try:
        from ultralytics import YOLO
        deps['ultralytics'] = True
    except ImportError:
        pass
    
    try:
        from pythonosc import udp_client
        deps['python_osc'] = True
    except ImportError:
        pass
    
    try:
        import netifaces
        deps['netifaces'] = True
    except ImportError:
        pass
    
    return deps

def main():
    """Main startup function when run directly"""
    parser = argparse.ArgumentParser(description='Unified Projection Mapping Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--no-reload', action='store_true', help='Disable auto-reload')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    
    args = parser.parse_args()
    
    print("🎯 Unified Projection Mapping Server")
    print("=" * 60)
    
    # Setup environment
    local_ip, cpu_count, total_memory_gb = setup_environment()
    
    # Check system resources
    resources = check_system_resources()
    if resources:
        print("💻 SYSTEM RESOURCES:")
        print(f"   CPU: {resources['cpu_count']} cores @ {resources['cpu_usage']:.1f}% usage")
        print(f"   RAM: {resources['memory_gb']:.1f}GB total, {resources['memory_available_gb']:.1f}GB available")
        print(f"   Disk: {resources['disk_free_gb']:.1f}GB free space")
    
    # Check GPU
    gpu_info = check_gpu_status()
    if gpu_info:
        print("\n🎮 GPU RESOURCES:")
        for gpu in gpu_info:
            print(f"   GPU {gpu['id']}: {gpu['name']}")
            print(f"           {gpu['free_mb']}MB free / {gpu['total_mb']}MB total")
    else:
        print("\n🖥️ No NVIDIA GPU detected")
    
    # Check dependencies
    deps = check_dependencies()
    print(f"\n📦 DEPENDENCIES:")
    print(f"   MediaPipe: {'✅' if deps['mediapipe'] else '❌'}")
    print(f"   YOLO: {'✅' if deps['ultralytics'] else '❌'}")
    print(f"   OSC: {'✅' if deps['python_osc'] else '❌'}")
    print(f"   Network: {'✅' if deps['netifaces'] else '❌'}")
    
    # Check stored images
    assets_dir = "app/assets"
    if os.path.exists(assets_dir):
        images = [f for f in os.listdir(assets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"\n📁 STORED IMAGES ({len(images)}):")
        for img in images:
            print(f"   📸 {img}")
    else:
        print("\n⚠️ No stored images found in app/assets/")
    
    print(f"\n🌐 NETWORK CONFIGURATION:")
    print(f"   📍 Local IP: {local_ip}")
    print(f"   🎵 OSC Port: 8001 (TouchDesigner, Max/MSP)")
    print(f"   🔗 TCP Port: 8002 (persistent connections)")
    print(f"   📦 UDP Port: 8003 (broadcast)")
    
    print(f"\n🚀 [RESOURCES] Configured for FULL system utilization:")
    print(f"   🖥️ CPU: ALL {cpu_count} cores")
    print(f"   💾 RAM: ALL {total_memory_gb:.1f}GB available")
    print(f"   🎮 GPU: FULL utilization (no memory limits)")
    print(f"   ⚡ Mode: Maximum performance")
    
    print(f"\n🎯 FEATURES ENABLED:")
    print(f"   ✅ Simple object matching (stored images vs camera)")
    print(f"   ✅ Real-time processing (MediaPipe, YOLO, OpenCV)")
    print(f"   ✅ Dynamic resource allocation (GPU/CPU auto)")
    print(f"   ✅ Network broadcasting (OSC, TCP, UDP)")
    print(f"   ✅ Detailed logging (terminal output)")
    
    print(f"\n🚀 STARTING SERVER:")
    print(f"   🌐 Web Interface: http://{local_ip}:8000")
    print(f"   📊 Network Info: http://{local_ip}:8000/api/network")
    print(f"   📡 Broadcasting to LAN network")
    print(f"   🎬 Real-time object detection ready")
    
    print("\n📋 CLIENT EXAMPLES:")
    print(f"   OSC: python network_clients/osc_client_example.py")
    print(f"   TCP: python network_clients/tcp_client_example.py --host {local_ip}")
    print(f"   UDP: python network_clients/udp_client_example.py")
    
    print("=" * 60)
    
    # Start the server
    try:
        cmd = [
            sys.executable, '-m', 'uvicorn',
            'app.main:app',
            '--host', args.host,
            '--port', str(args.port),
            '--workers', str(args.workers)
        ]
        
        if not args.no_reload and args.workers == 1:
            cmd.append('--reload')
        
        # Add performance optimizations
        cmd.extend([
            '--loop', 'uvloop',
            '--http', 'httptools',
            '--access-log',
            '--use-colors'
        ])
        
        subprocess.run(cmd, env=os.environ.copy())
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
