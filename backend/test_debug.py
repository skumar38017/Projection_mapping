#!/usr/bin/env python3
"""
Debug script to test individual components
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all imports work"""
    print("🔍 Testing Imports...")
    
    try:
        import cv2
        print("   ✅ OpenCV imported successfully")
    except Exception as e:
        print(f"   ❌ OpenCV import failed: {e}")
        return False
    
    try:
        from app.config import settings
        print("   ✅ Config imported successfully")
        print(f"      - Assets dir: {settings.ASSETS_DIR}")
        print(f"      - Static dir: {settings.STATIC_DIR}")
    except Exception as e:
        print(f"   ❌ Config import failed: {e}")
        return False
    
    try:
        from app.webrtc_signaling import CameraManager, get_reference_images_list
        print("   ✅ WebRTC signaling imported successfully")
    except Exception as e:
        print(f"   ❌ WebRTC signaling import failed: {e}")
        return False
    
    return True

def test_camera_basic():
    """Test basic camera functionality"""
    print("\n📹 Testing Basic Camera...")
    
    import cv2
    
    # Test camera 0
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"   ✅ Camera 0 working: {frame.shape}")
            cap.release()
            return True
        else:
            print("   ❌ Camera 0: Can't read frame")
    else:
        print("   ❌ Camera 0: Can't open")
    
    cap.release()
    return False

def test_reference_images():
    """Test reference image loading"""
    print("\n🖼️  Testing Reference Images...")
    
    from app.webrtc_signaling import get_reference_images_list
    
    try:
        images = get_reference_images_list()
        print(f"   ✅ Loaded {len(images)} reference images")
        
        for img in images:
            print(f"      - {img['filename']}: {img['width']}x{img['height']}")
        
        return len(images) > 0
    except Exception as e:
        print(f"   ❌ Reference image loading failed: {e}")
        return False

def test_camera_manager():
    """Test camera manager"""
    print("\n🎥 Testing Camera Manager...")
    
    try:
        from app.webrtc_signaling import CameraManager
        
        camera_manager = CameraManager()
        cameras = camera_manager.get_available_cameras()
        
        print(f"   ✅ Camera Manager found {len(cameras)} cameras")
        for cam in cameras:
            print(f"      - {cam['name']}: {cam['width']}x{cam['height']} @ {cam['fps']}fps")
        
        return len(cameras) > 0
    except Exception as e:
        print(f"   ❌ Camera Manager failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Debug Tests\n")
    
    tests = [
        ("Imports", test_imports),
        ("Basic Camera", test_camera_basic),
        ("Reference Images", test_reference_images),
        ("Camera Manager", test_camera_manager)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n📊 Test Results:")
    print("=" * 40)
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

if __name__ == "__main__":
    main()
