#!/usr/bin/env python3
"""
TCP Client Example - Receive object detection data via TCP
Persistent connection for reliable data streaming
"""

import socket
import json
import argparse
import threading

class TCPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
    
    def connect(self):
        """Connect to TCP server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.running = True
            print(f"✅ Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def listen(self):
        """Listen for incoming data"""
        buffer = ""
        
        while self.running:
            try:
                data = self.socket.recv(1024).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # Process complete JSON messages (separated by newlines)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            message = json.loads(line)
                            self.handle_message(message)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ JSON decode error: {e}")
                
            except Exception as e:
                if self.running:
                    print(f"💥 Receive error: {e}")
                break
        
        print("🔌 Connection closed")
    
    def handle_message(self, message):
        """Handle received message"""
        try:
            timestamp = message.get('timestamp', 'Unknown')
            source_ip = message.get('local_ip', 'Unknown')
            result = message.get('detection_result', {})
            frame_info = message.get('frame_info', {})
            
            print(f"\n📡 [TCP] Received detection data")
            print(f"🕐 Time: {timestamp}")
            print(f"📍 From: {source_ip}")
            
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
        """Stop the client"""
        self.running = False
        if self.socket:
            self.socket.close()

def main():
    parser = argparse.ArgumentParser(description='TCP Client for Object Detection')
    parser.add_argument('--host', default='192.168.1.100', help='Server IP address')
    parser.add_argument('--port', type=int, default=8002, help='Server port')
    args = parser.parse_args()

    print("🔗 TCP Client for Object Detection")
    print("=" * 50)
    print(f"📡 Connecting to {args.host}:{args.port}")
    print("🎯 Waiting for detection data...")
    print("=" * 50)

    client = TCPClient(args.host, args.port)
    
    if client.connect():
        try:
            client.listen()
        except KeyboardInterrupt:
            print("\n🛑 TCP Client stopped")
        finally:
            client.stop()
    else:
        print("❌ Failed to connect to server")

if __name__ == "__main__":
    main()
