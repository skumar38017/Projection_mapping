#!/usr/bin/env python3
"""
UDP Client Example - Receive object detection data via UDP broadcast
Perfect for multiple devices receiving the same data
"""

import socket
import json
import argparse

class UDPClient:
    def __init__(self, port):
        self.port = port
        self.socket = None
        self.running = False
    
    def start(self):
        """Start UDP listener"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('', self.port))  # Listen on all interfaces
            self.running = True
            print(f"✅ UDP listener started on port {self.port}")
            return True
        except Exception as e:
            print(f"❌ Failed to start UDP listener: {e}")
            return False
    
    def listen(self):
        """Listen for UDP broadcasts"""
        while self.running:
            try:
                data, address = self.socket.recvfrom(4096)
                message = json.loads(data.decode('utf-8'))
                self.handle_message(message, address)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode error: {e}")
            except Exception as e:
                if self.running:
                    print(f"💥 Receive error: {e}")
                break
        
        print("🔌 UDP listener stopped")
    
    def handle_message(self, message, sender_address):
        """Handle received UDP message"""
        try:
            timestamp = message.get('timestamp', 'Unknown')
            source_ip = message.get('local_ip', 'Unknown')
            result = message.get('detection_result', {})
            frame_info = message.get('frame_info', {})
            
            print(f"\n📦 [UDP] Received from {sender_address[0]}:{sender_address[1]}")
            print(f"🕐 Time: {timestamp}")
            print(f"📍 Source: {source_ip}")
            
            if result.get('success'):
                print(f"✅ MATCH FOUND!")
                print(f"   📸 Image: {result.get('matched_image', 'Unknown')}")
                print(f"   📊 Score: {result.get('match_score', 0):.3f}")
                print(f"   🔗 Matches: {result.get('matches_count', 0)}")
                print(f"   🎯 Confidence: {result.get('details', {}).get('confidence', 0):.1f}%")
            else:
                print(f"❌ NO MATCH")
            
            print(f"⏱️ Processing: {result.get('processing_time', 0)*1000:.1f}ms")
            
            if frame_info:
                print(f"📷 Frame: {frame_info.get('frame_shape', 'Unknown')}")
                print(f"🔍 Features: {frame_info.get('features_extracted', 0)}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"⚠️ Message handling error: {e}")
    
    def stop(self):
        """Stop the UDP listener"""
        self.running = False
        if self.socket:
            self.socket.close()

def main():
    parser = argparse.ArgumentParser(description='UDP Client for Object Detection')
    parser.add_argument('--port', type=int, default=8003, help='UDP port to listen on')
    args = parser.parse_args()

    print("📦 UDP Client for Object Detection")
    print("=" * 50)
    print(f"📡 Listening for broadcasts on port {args.port}")
    print("🎯 Waiting for detection data...")
    print("=" * 50)

    client = UDPClient(args.port)
    
    if client.start():
        try:
            client.listen()
        except KeyboardInterrupt:
            print("\n🛑 UDP Client stopped")
        finally:
            client.stop()
    else:
        print("❌ Failed to start UDP listener")

if __name__ == "__main__":
    main()
