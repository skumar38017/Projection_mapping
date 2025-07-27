#!/usr/bin/env python3
"""
Random Object Verification System - Server Startup Script
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/server.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'opencv-python', 'numpy', 
        'tensorflow', 'torch', 'torchvision', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'app/debug', 'app/assets', 'app/static']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def main():
    """Main startup function"""
    logger.info("🎯 Starting Random Object Verification System")
    
    # Create necessary directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check for assets
    assets_dir = Path("app/assets")
    image_files = list(assets_dir.glob("*.jpg")) + list(assets_dir.glob("*.jpeg")) + list(assets_dir.glob("*.png"))
    
    if not image_files:
        logger.warning("⚠️  No reference images found in app/assets/")
        logger.warning("   Add some .jpg, .jpeg, or .png files to the assets directory for object verification")
    else:
        logger.info(f"📁 Found {len(image_files)} reference images")
        for img in image_files:
            logger.info(f"   - {img.name}")
    
    # Start the server
    logger.info("🚀 Starting FastAPI server...")
    logger.info("📱 Open your browser and go to: http://localhost:8000")
    logger.info("🛑 Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
