"""
Network Broadcaster - Send object detection data across LAN
Supports OSC, TCP, UDP protocols for projection mapping and other applications
"""

import socket
import json
import threading
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import struct
import netifaces
import asyncio

# OSC Protocol support
try:
    from pythonosc import udp_client, osc_message_builder
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

logger = logging.getLogger(__name__)

class NetworkBroadcaster:
    def __init__(self):
        self.enabled = True
        self.local_ip = self._get_local_ip()
        self.broadcast_ip = self._get_broadcast_ip()
        
        # Protocol clients/servers
        self.osc_client = None
        self.tcp_server = None
        self.udp_socket = None
        self.tcp_clients = []
        
        # Configuration
        self.osc_port = 8001
        self.tcp_port = 8002
        self.udp_port = 8003
        self.broadcast_port = 8004
        
        # Initialize protocols
        self._init_osc()
        self._init_tcp_server()
        self._init_udp()
        
        print(f"\n🌐 [NETWORK] Network Broadcaster initialized")
        print(f"   📍 Local IP: {self.local_ip}")
        print(f"   📡 Broadcast IP: {self.broadcast_ip}")
        print(f"   🎵 OSC Port: {self.osc_port}")
        print(f"   🔗 TCP Port: {self.tcp_port}")
        print(f"   📦 UDP Port: {self.udp_port}")
        
    def _get_local_ip(self) -> str:
        """Get local machine IP address"""
        try:
            # Get default gateway interface
            gateways = netifaces.gateways()
            default_interface = gateways['default'][netifaces.AF_INET][1]
            
            # Get IP of default interface
            addresses = netifaces.ifaddresses(default_interface)
            local_ip = addresses[netifaces.AF_INET][0]['addr']
            
            return local_ip
        except:
            # Fallback method
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                return local_ip
            except:
                return "127.0.0.1"
    
    def _get_broadcast_ip(self) -> str:
        """Get broadcast IP for LAN"""
        try:
            # Get network info
            gateways = netifaces.gateways()
            default_interface = gateways['default'][netifaces.AF_INET][1]
            addresses = netifaces.ifaddresses(default_interface)
            
            # Calculate broadcast address
            ip = addresses[netifaces.AF_INET][0]['addr']
            netmask = addresses[netifaces.AF_INET][0]['netmask']
            
            # Convert to broadcast IP
            ip_parts = ip.split('.')
            mask_parts = netmask.split('.')
            
            broadcast_parts = []
            for i in range(4):
                ip_byte = int(ip_parts[i])
                mask_byte = int(mask_parts[i])
                broadcast_byte = ip_byte | (255 - mask_byte)
                broadcast_parts.append(str(broadcast_byte))
            
            return '.'.join(broadcast_parts)
        except:
            # Default broadcast
            return "255.255.255.255"
    
    def _init_osc(self):
        """Initialize OSC client"""
        if not OSC_AVAILABLE:
            print("⚠️ [OSC] python-osc not available. Install with: pip install python-osc")
            return
        
        try:
            # Create OSC client for broadcasting
            self.osc_client = udp_client.SimpleUDPClient(self.broadcast_ip, self.osc_port)
            print(f"✅ [OSC] Client ready on {self.broadcast_ip}:{self.osc_port}")
        except Exception as e:
            print(f"❌ [OSC] Failed to initialize: {e}")
    
    def _init_tcp_server(self):
        """Initialize TCP server for persistent connections"""
        try:
            self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_server.bind((self.local_ip, self.tcp_port))
            self.tcp_server.listen(5)
            
            # Start TCP server thread
            tcp_thread = threading.Thread(target=self._tcp_server_loop, daemon=True)
            tcp_thread.start()
            
            print(f"✅ [TCP] Server listening on {self.local_ip}:{self.tcp_port}")
        except Exception as e:
            print(f"❌ [TCP] Failed to initialize: {e}")
    
    def _init_udp(self):
        """Initialize UDP socket for broadcasting"""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            print(f"✅ [UDP] Socket ready for broadcast to {self.broadcast_ip}:{self.udp_port}")
        except Exception as e:
            print(f"❌ [UDP] Failed to initialize: {e}")
    
    def _tcp_server_loop(self):
        """TCP server loop to accept connections"""
        while self.enabled:
            try:
                client_socket, address = self.tcp_server.accept()
                print(f"🔗 [TCP] New client connected: {address}")
                
                # Add to client list
                self.tcp_clients.append({
                    'socket': client_socket,
                    'address': address,
                    'connected_at': datetime.now()
                })
                
                # Start client handler thread
                client_thread = threading.Thread(
                    target=self._handle_tcp_client, 
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.enabled:
                    logger.error(f"TCP server error: {e}")
                break
    
    def _handle_tcp_client(self, client_socket, address):
        """Handle individual TCP client"""
        try:
            while self.enabled:
                # Keep connection alive
                time.sleep(1)
        except Exception as e:
            logger.debug(f"TCP client {address} disconnected: {e}")
        finally:
            # Remove from client list
            self.tcp_clients = [c for c in self.tcp_clients if c['address'] != address]
            client_socket.close()
            print(f"❌ [TCP] Client disconnected: {address}")
    
    def broadcast_detection_result(self, result: Dict[str, Any], frame_data: Optional[Dict] = None):
        """Broadcast detection result to all network protocols"""
        if not self.enabled:
            return
        
        try:
            # Prepare broadcast data
            broadcast_data = {
                'timestamp': datetime.now().isoformat(),
                'source': 'projection_mapping',
                'local_ip': self.local_ip,
                'detection_result': result,
                'frame_info': frame_data or {}
            }
            
            print(f"\n📡 [BROADCAST] Sending detection result to network...")
            print(f"   🎯 Match: {result.get('success', False)}")
            if result.get('success'):
                print(f"   📸 Image: {result.get('matched_image', 'unknown')}")
                print(f"   📊 Score: {result.get('match_score', 0):.3f}")
            
            # Send via all protocols
            self._send_osc(broadcast_data)
            self._send_tcp(broadcast_data)
            self._send_udp(broadcast_data)
            
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
    
    def _send_osc(self, data: Dict[str, Any]):
        """Send data via OSC protocol"""
        if not self.osc_client or not OSC_AVAILABLE:
            return
        
        try:
            result = data['detection_result']
            
            # Send basic detection info
            self.osc_client.send_message("/detection/success", result.get('success', False))
            
            if result.get('success'):
                self.osc_client.send_message("/detection/image", result.get('matched_image', ''))
                self.osc_client.send_message("/detection/score", result.get('match_score', 0.0))
                self.osc_client.send_message("/detection/matches", result.get('matches_count', 0))
                self.osc_client.send_message("/detection/confidence", result.get('details', {}).get('confidence', 0.0))
            
            # Send timing info
            self.osc_client.send_message("/detection/processing_time", result.get('processing_time', 0.0))
            self.osc_client.send_message("/detection/timestamp", data['timestamp'])
            
            # Send source info
            self.osc_client.send_message("/source/ip", self.local_ip)
            
            print(f"   🎵 [OSC] Sent to {self.broadcast_ip}:{self.osc_port}")
            
        except Exception as e:
            logger.debug(f"OSC send error: {e}")
    
    def _send_tcp(self, data: Dict[str, Any]):
        """Send data via TCP to connected clients"""
        if not self.tcp_clients:
            return
        
        try:
            # Prepare JSON message
            message = json.dumps(data) + '\n'
            message_bytes = message.encode('utf-8')
            
            # Send to all connected clients
            disconnected_clients = []
            
            for client in self.tcp_clients:
                try:
                    client['socket'].send(message_bytes)
                except Exception as e:
                    logger.debug(f"TCP client send error: {e}")
                    disconnected_clients.append(client)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                self.tcp_clients.remove(client)
                client['socket'].close()
            
            if len(self.tcp_clients) > 0:
                print(f"   🔗 [TCP] Sent to {len(self.tcp_clients)} clients")
            
        except Exception as e:
            logger.debug(f"TCP send error: {e}")
    
    def _send_udp(self, data: Dict[str, Any]):
        """Send data via UDP broadcast"""
        if not self.udp_socket:
            return
        
        try:
            # Prepare JSON message
            message = json.dumps(data)
            message_bytes = message.encode('utf-8')
            
            # Send UDP broadcast
            self.udp_socket.sendto(message_bytes, (self.broadcast_ip, self.udp_port))
            print(f"   📦 [UDP] Broadcast to {self.broadcast_ip}:{self.udp_port}")
            
        except Exception as e:
            logger.debug(f"UDP send error: {e}")
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get network configuration info"""
        return {
            'local_ip': self.local_ip,
            'broadcast_ip': self.broadcast_ip,
            'ports': {
                'osc': self.osc_port,
                'tcp': self.tcp_port,
                'udp': self.udp_port
            },
            'protocols': {
                'osc_available': OSC_AVAILABLE,
                'tcp_clients': len(self.tcp_clients),
                'udp_ready': self.udp_socket is not None
            },
            'tcp_clients': [
                {
                    'address': client['address'],
                    'connected_at': client['connected_at'].isoformat()
                }
                for client in self.tcp_clients
            ]
        }
    
    def stop(self):
        """Stop all network services"""
        print(f"\n🛑 [NETWORK] Stopping network broadcaster...")
        self.enabled = False
        
        # Close TCP server
        if self.tcp_server:
            self.tcp_server.close()
        
        # Close TCP clients
        for client in self.tcp_clients:
            client['socket'].close()
        
        # Close UDP socket
        if self.udp_socket:
            self.udp_socket.close()
        
        print(f"✅ [NETWORK] Network broadcaster stopped")

# Global network broadcaster instance
network_broadcaster = None

def get_network_broadcaster() -> NetworkBroadcaster:
    """Get or create network broadcaster instance"""
    global network_broadcaster
    if network_broadcaster is None:
        network_broadcaster = NetworkBroadcaster()
    return network_broadcaster

def broadcast_to_network(result: Dict[str, Any], frame_data: Optional[Dict] = None):
    """Broadcast detection result to network"""
    broadcaster = get_network_broadcaster()
    broadcaster.broadcast_detection_result(result, frame_data)
