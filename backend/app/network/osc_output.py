# ~/app/network/osc_output.py
from pythonosc import udp_client
import logging
from typing import Dict, Any, Union, List
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat 

logger = logging.getLogger(__name__)

class OSCOutput:
    def __init__(self, ip: str = "localhost", port: int = 5005, debug: bool = True):
        """
        Initialize OSC output handler
        :param ip: Target IP address
        :param port: Target OSC port
        """
        self.ip = ip
        self.port = port
        self.debug = debug
        self.executor = ThreadPoolExecutor(max_workers=2)
        try:
            self.client = udp_client.SimpleUDPClient(ip, port)
            logger.info(f"Initialized OSC client for {ip}:{port}")
        except Exception as e:
            logger.error(f"Failed to initialize OSC client: {str(e)}")
            self.client = None

    def send_data(self, address: str, data: Union[Dict[str, Any], List[Any], Any]):
        """Send data via OSC"""
        if not self.client:
            logger.warning("Attempting to reconnect OSC client")
            try:
                self.client = udp_client.SimpleUDPClient(self.ip, self.port)
            except Exception as e:
                logger.error(f"OSC reconnect failed: {str(e)}")
                return
            
        try:
            if self.debug:
                print("\nOSC DEBUG OUTPUT:")
                print(f"Target: {self.ip}:{self.port}")
                print(f"Address: {address}")
                print("Data:")
                if isinstance(data, dict):
                    for k, v in data.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  {data}")
                print("-"*40)
            
            if isinstance(data, dict):
                # For dictionaries, send each key-value pair as separate OSC messages
                for key, value in data.items():
                    self._send_single_message(f"{address}/{key}", value)
            elif isinstance(data, list):
                # For lists, send as an OSC bundle
                self.client.send_message(address, data)
            else:
                # For single values, send directly
                self._send_single_message(address, data)
            logger.debug(f"Sent OSC data to {address}")
        except Exception as e:
            logger.error(f"Failed to send OSC data: {str(e)}")

    def _send_single_message(self, address: str, value: Any):
        """Handle type conversion for single OSC messages"""
        # Convert numpy types to native Python types
        if hasattr(value, 'item'):
            value = value.item()
        self.client.send_message(address, value)

    def async_send(self, address: str, data: Union[Dict[str, Any], List[Any], Any]):
        """Send data asynchronously"""
        if self.client:
            self.executor.submit(self.send_data, address, data)
        
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("OSC output closed")