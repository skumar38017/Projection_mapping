#!/usr/bin/env python3
"""
Test Enhanced Object Matching System
Demonstrates the new hybrid approach combining:
- Traditional computer vision (ORB + SIFT)
- Deep learning embeddings (CLIP, ResNet, MobileNet, EfficientNet)
- Vector similarity search (Pinecone + FAISS)
- 60% threshold decision making
"""

import cv2
import numpy as np
import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.enhanced_matcher import get_enhanced_matcher

def test_enhanced_matching():
    """Test the enhanced matching system with multiple approaches"""
    print("🎯 Enhanced Real-Time 3D Object Detection & Matching System")
    print("=" * 70)
    
    # Initialize enhanced matcher
    matcher = get_enhanced_matcher()
    
    # Get system info
    info = matcher.get_stored_images_info()
    print(f"📷 Stored Images: {info['stored_images']}")
    print(f"🎯 Threshold: {info['threshold']}%")
    print(f"🔧 Traditional Methods: {info['matching_methods'][0]}")
    print(f"🧠 Deep Learning Models: {info['embedding_models']}")
    print(f"☁️ Pinecone Available: {info['vector_search']['pinecone_available']}")
    print(f"💾 FAISS Available: {info['vector_search']['faiss_available']}")
    print()
    
    # Load test image
    watch_path = "app/assets/watch.jpeg"
    if not os.path.exists(watch_path):
        print(f"❌ Test image not found: {watch_path}")
        return
    
    # Test 1: Perfect match (same image)
    print("🧪 TEST 1: Perfect Match (Enhanced Hybrid Approach)")
    print("-" * 50)
    test_image = cv2.imread(watch_path)
    if test_image is not None:
        result, processed_frame = matcher.match_with_stored_images(test_image)
        
        print(f"\n📊 FINAL RESULT:")
        print(f"   Match Result: {result['match_result']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   Threshold Met: {result['threshold_met']}")
        print(f"   Method: {result['method']}")
        print(f"   Processing Time: {result['processing_time_ms']:.1f}ms")
        
        if 'details' in result:
            details = result['details']
            print(f"   Traditional Score: {details['traditional_score']:.1f}%")
            print(f"   Vector Score: {details['vector_score']:.1f}%")
        print()
    
    # Test 2: Modified image (brightness + noise)
    print("🧪 TEST 2: Modified Image (brightness + noise)")
    print("-" * 50)
    if test_image is not None:
        # Apply modifications
        modified_image = cv2.convertScaleAbs(test_image, alpha=1.3, beta=30)
        noise = np.random.normal(0, 25, modified_image.shape).astype(np.uint8)
        modified_image = cv2.add(modified_image, noise)
        
        result, processed_frame = matcher.match_with_stored_images(modified_image)
        
        print(f"\n📊 FINAL RESULT:")
        print(f"   Match Result: {result['match_result']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   Threshold Met: {result['threshold_met']}")
        print(f"   Method: {result['method']}")
        print(f"   Processing Time: {result['processing_time_ms']:.1f}ms")
        
        if 'details' in result:
            details = result['details']
            print(f"   Traditional Score: {details['traditional_score']:.1f}%")
            print(f"   Vector Score: {details['vector_score']:.1f}%")
        print()
    
    # Test 3: Random noise (should not match)
    print("🧪 TEST 3: Random Noise (should not match)")
    print("-" * 50)
    noise_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result, processed_frame = matcher.match_with_stored_images(noise_image)
    
    print(f"\n📊 FINAL RESULT:")
    print(f"   Match Result: {result['match_result']}")
    print(f"   Confidence: {result['confidence']:.1f}%")
    print(f"   Threshold Met: {result['threshold_met']}")
    print(f"   Method: {result['method']}")
    print(f"   Processing Time: {result['processing_time_ms']:.1f}ms")
    
    if 'details' in result:
        details = result['details']
        print(f"   Traditional Score: {details['traditional_score']:.1f}%")
        print(f"   Vector Score: {details['vector_score']:.1f}%")
    print()
    
    # Test 4: Rotated image
    print("🧪 TEST 4: Rotated Image (45 degrees)")
    print("-" * 50)
    if test_image is not None:
        # Rotate image 45 degrees
        h, w = test_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
        rotated_image = cv2.warpAffine(test_image, rotation_matrix, (w, h))
        
        result, processed_frame = matcher.match_with_stored_images(rotated_image)
        
        print(f"\n📊 FINAL RESULT:")
        print(f"   Match Result: {result['match_result']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   Threshold Met: {result['threshold_met']}")
        print(f"   Method: {result['method']}")
        print(f"   Processing Time: {result['processing_time_ms']:.1f}ms")
        
        if 'details' in result:
            details = result['details']
            print(f"   Traditional Score: {details['traditional_score']:.1f}%")
            print(f"   Vector Score: {details['vector_score']:.1f}%")
        print()
    
    print("✅ Enhanced matching testing completed!")
    print("\n📋 SYSTEM CAPABILITIES:")
    print("   🔍 Traditional Computer Vision:")
    print("     • ORB feature detection and matching")
    print("     • SIFT feature detection and matching")
    print("     • Robust to lighting and minor transformations")
    print()
    print("   🧠 Deep Learning Embeddings:")
    print("     • CLIP: Vision Transformer (512D)")
    print("     • ResNet50: Convolutional features (2048D)")
    print("     • MobileNetV2: Lightweight features (1280D)")
    print("     • EfficientNet-B0: Balanced features (1280D)")
    print()
    print("   🔍 Vector Similarity Search:")
    print("     • Pinecone: Cloud-based vector database")
    print("     • FAISS: Local high-performance search")
    print("     • Cosine similarity matching")
    print()
    print("   ⚖️ Hybrid Decision Making:")
    print("     • Traditional features: 30% weight")
    print("     • Deep learning: 70% weight")
    print("     • 60% threshold for final decision")
    print("     • Real-time network broadcasting")
    print()
    print("   📡 Network Broadcasting:")
    print("     • OSC (Port 8001): TouchDesigner, Max/MSP")
    print("     • TCP (Port 8002): Persistent connections")
    print("     • UDP (Port 8003): Broadcast messages")
    print("     • Socket.IO (Port 8000): Web clients")

if __name__ == "__main__":
    test_enhanced_matching()
