#!/usr/bin/env python3
"""
System Test Script for Random Object Verification System
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
        print(f"   TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    return True

def test_cameras():
    """Test camera detection"""
    print("\n📹 Testing camera detection...")
    
    cameras_found = []
    for i in range(5):  # Test first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cameras_found.append(i)
                print(f"✅ Camera {i} detected and working")
            cap.release()
    
    if cameras_found:
        print(f"📹 Found {len(cameras_found)} working cameras: {cameras_found}")
        return True
    else:
        print("❌ No working cameras found")
        return False

def test_reference_images():
    """Test reference images loading"""
    print("\n🖼️  Testing reference images...")
    
    assets_dir = Path("app/assets")
    if not assets_dir.exists():
        print("❌ Assets directory not found")
        return False
    
    image_files = list(assets_dir.glob("*.jpg")) + list(assets_dir.glob("*.jpeg")) + list(assets_dir.glob("*.png"))
    
    if not image_files:
        print("⚠️  No reference images found in assets directory")
        print("   Add some .jpg, .jpeg, or .png files to app/assets/ for testing")
        return False
    
    working_images = []
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                working_images.append(img_path.name)
                print(f"✅ {img_path.name} loaded successfully ({img.shape})")
            else:
                print(f"❌ Failed to load {img_path.name}")
        except Exception as e:
            print(f"❌ Error loading {img_path.name}: {e}")
    
    print(f"🖼️  {len(working_images)} reference images ready for use")
    return len(working_images) > 0

def test_object_detector():
    """Test object detector initialization"""
    print("\n🔍 Testing object detector...")
    
    try:
        from app.object_detector import ObjectDetector
        detector = ObjectDetector()
        
        print("✅ ObjectDetector initialized successfully")
        
        # Test with a simple test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Test feature extraction
        features = detector.extract_features(test_img)
        if features is not None:
            print("✅ Feature extraction working")
        else:
            print("⚠️  Feature extraction returned None (may be normal for test image)")
        
        # Test deep features
        if detector.deep_feature_extractor is not None:
            deep_features = detector.extract_deep_features(test_img)
            if deep_features is not None:
                print("✅ Deep feature extraction working")
            else:
                print("⚠️  Deep feature extraction returned None")
        
        return True
        
    except Exception as e:
        print(f"❌ ObjectDetector test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n⚙️  Testing configuration...")
    
    try:
        from app.config import settings
        print("✅ Configuration loaded successfully")
        print(f"   Project: {settings.PROJECT_NAME}")
        print(f"   Assets dir: {settings.ASSETS_DIR}")
        print(f"   Feature threshold: {settings.FEATURE_MATCH_THRESHOLD}")
        print(f"   Deep threshold: {settings.DEEP_MATCH_THRESHOLD}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 Random Object Verification System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Reference Images", test_reference_images),
        ("Cameras", test_cameras),
        ("Object Detector", test_object_detector),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready to use.")
        print("🚀 Run 'python start_server.py' to start the server")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("💡 Refer to the README.md for troubleshooting help")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
