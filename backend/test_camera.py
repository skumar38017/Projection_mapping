#!/usr/bin/env python3
"""
Test script to check camera detection and basic functionality
"""

import cv2
import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.config import settings
from app.webrtc_signaling import CameraManager, get_reference_images_list

def test_camera_detection():
    """Test camera detection"""
    print("🔍 Testing Camera Detection...")
    
    # Test basic OpenCV camera detection
    print("\n1. Basic OpenCV Camera Test:")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                print(f"   ✅ Camera {i}: {width}x{height} @ {fps}fps")
            else:
                print(f"   ❌ Camera {i}: Can't read frame")
            cap.release()
        else:
            print(f"   ❌ Camera {i}: Can't open")
    
    # Test CameraManager
    print("\n2. CameraManager Test:")
    camera_manager = CameraManager()
    cameras = camera_manager.get_available_cameras()
    
    if cameras:
        print(f"   ✅ Found {len(cameras)} cameras:")
        for cam in cameras:
            print(f"      - {cam['name']}: {cam['width']}x{cam['height']} @ {cam['fps']}fps")
    else:
        print("   ❌ No cameras found by CameraManager")
    
    return len(cameras) > 0

def test_reference_images():
    """Test reference image loading"""
    print("\n🖼️  Testing Reference Images...")
    
    # Check assets directory
    print(f"Assets directory: {settings.ASSETS_DIR}")
    print(f"Directory exists: {settings.ASSETS_DIR.exists()}")
    
    if settings.ASSETS_DIR.exists():
        files = list(settings.ASSETS_DIR.iterdir())
        print(f"Files in directory: {len(files)}")
        for file in files:
            print(f"   - {file.name}")
    
    # Test reference image loading
    images = get_reference_images_list()
    print(f"\n✅ Loaded {len(images)} reference images:")
    for img in images:
        print(f"   - {img['filename']}: {img['width']}x{img['height']} ({img['size']} bytes)")
    
    return len(images) > 0

def test_object_detector():
    """Test object detector initialization"""
    print("\n🤖 Testing Object Detector...")
    
    try:
        from app.object_detector import ObjectDetector
        detector = ObjectDetector()
        
        print(f"   ✅ Feature extractor: {'✓' if detector.feature_extractor else '✗'}")
        print(f"   ✅ Deep feature extractor: {'✓' if detector.deep_feature_extractor else '✗'}")
        print(f"   ✅ Detection model: {'✓' if detector.detection_model else '✗'}")
        print(f"   ✅ Reference features: {len(detector.ref_features)}")
        print(f"   ✅ Deep reference features: {len(detector.ref_deep_features)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error initializing detector: {e}")
        return False

def test_video_devices():
    """Test video devices using v4l2"""
    print("\n📹 Testing Video Devices (v4l2)...")
    
    try:
        # List video devices
        video_devices = []
        for i in range(10):
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                video_devices.append(device_path)
        
        print(f"   Found {len(video_devices)} video devices:")
        for device in video_devices:
            print(f"      - {device}")
        
        return len(video_devices) > 0
    except Exception as e:
        print(f"   ❌ Error checking video devices: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running System Tests for Projection Mapping Backend\n")
    
    results = {
        "camera_detection": test_camera_detection(),
        "reference_images": test_reference_images(),
        "object_detector": test_object_detector(),
        "video_devices": test_video_devices()
    }
    
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if not all_passed:
        print("\n🔧 Troubleshooting Tips:")
        if not results["camera_detection"]:
            print("   - Check if camera is connected and not used by another application")
            print("   - Try: sudo modprobe uvcvideo")
            print("   - Check permissions: ls -la /dev/video*")
        
        if not results["reference_images"]:
            print("   - Add reference images to app/assets/ directory")
            print("   - Supported formats: .png, .jpg, .jpeg, .bmp, .tiff, .webp")
        
        if not results["object_detector"]:
            print("   - Check if all required packages are installed")
            print("   - Try: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
