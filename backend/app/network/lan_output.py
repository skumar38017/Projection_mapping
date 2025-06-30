# ~/app/network/lan_output.py
import socket
import json
import logging
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat 

logger = logging.getLogger(__name__)

class LANOutput:
    def __init__(self, host: str = '255.255.255.255', port: int = 5000, protocol: str = 'udp', debug: bool = True):
        """
        Initialize LAN output handler
        :param host: Target host or broadcast address
        :param port: Target port
        :param protocol: 'udp' or 'tcp'
        """
        self.host = host
        self.port = port
        self.protocol = protocol.lower()
        self.debug = debug
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.socket = None
        
        if self.protocol == 'tcp':
            self._setup_tcp_connection()
        
    def _setup_tcp_connection(self):
        """Setup persistent TCP connection"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            logger.info(f"TCP connection established to {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to establish TCP connection: {str(e)}")
            self.socket = None
            
    def send_data(self, data: Dict[str, Any]):
        """Send data over LAN"""
        if self.protocol == 'tcp' and self.socket is None:
            self._setup_tcp_connection()
        try:
            json_data = json.dumps(data, indent=2).encode('utf-8')
            if self.debug:
                print("\n" + "="*40)
                print("LAN OUTPUT:")
                print(pformat(data))  # Pretty-print the dictionary
                print("="*40 + "\n")
            
            if self.protocol == 'udp' or self.protocol == 'osc':
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    s.sendto(json_data, (self.host, self.port))
            elif self.protocol == 'tcp' and self.socket:
                self.socket.sendall(json_data)
            
            logger.debug(f"Sent data to {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to send LAN data: {str(e)}")
            if self.protocol == 'tcp':
                self._reconnect_tcp()

    def _reconnect_tcp(self):
        """Attempt to reconnect TCP socket"""
        try:
            if self.socket:
                self.socket.close()
            self._setup_tcp_connection()
        except Exception as e:
            logger.error(f"TCP reconnection failed: {str(e)}")

    def async_send(self, data: Dict[str, Any]):
        """Send data asynchronously"""
        self.executor.submit(self.send_data, data)
        
    def close(self):
        """Clean up resources"""
        if self.socket:
            self.socket.close()
        self.executor.shutdown(wait=True)