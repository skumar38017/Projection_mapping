#!/usr/bin/env python3
"""
OSC Client Example - Receive object detection data via OSC
Perfect for projection mapping software like TouchDesigner, Max/MSP, etc.
"""

from pythonosc import dispatcher, server
import argparse

def detection_success_handler(unused_addr, success):
    """Handle detection success/failure"""
    status = "✅ MATCH FOUND" if success else "❌ NO MATCH"
    print(f"{status}")

def detection_image_handler(unused_addr, image_name):
    """Handle matched image name"""
    print(f"📸 Matched Image: {image_name}")

def detection_score_handler(unused_addr, score):
    """Handle match score"""
    print(f"📊 Match Score: {score:.3f} ({score*100:.1f}%)")

def detection_matches_handler(unused_addr, matches):
    """Handle number of feature matches"""
    print(f"🔗 Feature Matches: {matches}")

def detection_confidence_handler(unused_addr, confidence):
    """Handle confidence percentage"""
    print(f"🎯 Confidence: {confidence:.1f}%")

def detection_time_handler(unused_addr, processing_time):
    """Handle processing time"""
    print(f"⏱️ Processing Time: {processing_time*1000:.1f}ms")

def detection_timestamp_handler(unused_addr, timestamp):
    """Handle timestamp"""
    print(f"🕐 Timestamp: {timestamp}")

def source_ip_handler(unused_addr, ip):
    """Handle source IP"""
    print(f"📍 Source IP: {ip}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='OSC Client for Object Detection')
    parser.add_argument('--ip', default='0.0.0.0', help='IP to listen on')
    parser.add_argument('--port', type=int, default=8001, help='Port to listen on')
    args = parser.parse_args()

    print("🎵 OSC Client for Object Detection")
    print("=" * 50)
    print(f"📡 Listening on {args.ip}:{args.port}")
    print("🎯 Waiting for detection data...")
    print("=" * 50)

    # Create dispatcher and add handlers
    disp = dispatcher.Dispatcher()
    disp.map("/detection/success", detection_success_handler)
    disp.map("/detection/image", detection_image_handler)
    disp.map("/detection/score", detection_score_handler)
    disp.map("/detection/matches", detection_matches_handler)
    disp.map("/detection/confidence", detection_confidence_handler)
    disp.map("/detection/processing_time", detection_time_handler)
    disp.map("/detection/timestamp", detection_timestamp_handler)
    disp.map("/source/ip", source_ip_handler)

    # Start OSC server
    server_instance = server.osc.ThreadingOSCUDPServer((args.ip, args.port), disp)
    print(f"✅ OSC Server started on {args.ip}:{args.port}")
    
    try:
        server_instance.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 OSC Client stopped")

if __name__ == "__main__":
    main()
