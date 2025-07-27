"""
Dynamic Resource Monitor
Monitors system resources and adjusts processing parameters dynamically
"""

import psutil
import threading
import time
import logging
from typing import Dict, Any, Optional
import subprocess
import json

logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Resource metrics
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.gpu_usage = 0.0
        self.gpu_memory_usage = 0.0
        self.disk_usage = 0.0
        self.network_usage = {'sent': 0, 'recv': 0}
        
        # System limits (dynamic)
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = psutil.virtual_memory().available
        
        # Performance metrics
        self.frame_processing_times = []
        self.average_fps = 0.0
        self.target_fps = 30.0
        
        # Dynamic adjustment parameters
        self.auto_adjust = True
        self.performance_mode = "balanced"  # "performance", "balanced", "efficiency"
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("🔍 Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("🔍 Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._update_system_metrics()
                self._update_gpu_metrics()
                self._adjust_performance_parameters()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _update_system_metrics(self):
        """Update CPU, memory, disk, and network metrics"""
        try:
            # CPU usage
            self.cpu_usage = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage = memory.percent
            self.available_memory = memory.available
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.disk_usage = disk.percent
            
            # Network usage
            net_io = psutil.net_io_counters()
            self.network_usage = {
                'sent': net_io.bytes_sent,
                'recv': net_io.bytes_recv
            }
            
        except Exception as e:
            logger.debug(f"System metrics update error: {e}")
    
    def _update_gpu_metrics(self):
        """Update GPU usage metrics"""
        try:
            # Try to get GPU metrics using nvidia-smi
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    gpu_util, mem_used, mem_total = lines[0].split(', ')
                    self.gpu_usage = float(gpu_util)
                    self.gpu_memory_usage = (float(mem_used) / float(mem_total)) * 100
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
            # Fallback: try to get GPU info from PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_memory_usage = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
            except:
                pass
    
    def _adjust_performance_parameters(self):
        """Dynamically adjust performance parameters based on resource usage"""
        if not self.auto_adjust:
            return
        
        # Determine performance mode based on resource usage
        if self.cpu_usage > 80 or self.memory_usage > 85 or self.gpu_usage > 90:
            self.performance_mode = "efficiency"
        elif self.cpu_usage < 50 and self.memory_usage < 60 and self.gpu_usage < 60:
            self.performance_mode = "performance"
        else:
            self.performance_mode = "balanced"
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on available resources - use more resources"""
        if self.performance_mode == "performance":
            return 8  # Increased batch size
        elif self.performance_mode == "balanced":
            return 4  # Increased batch size
        else:  # efficiency
            return 2  # Still reasonable
    
    def get_optimal_resolution(self) -> tuple:
        """Get optimal camera resolution based on performance - higher resolutions"""
        if self.performance_mode == "performance":
            return (1920, 1080)  # Full HD for maximum quality
        elif self.performance_mode == "balanced":
            return (1280, 720)   # HD
        else:  # efficiency
            return (640, 480)    # VGA
    
    def get_optimal_fps(self) -> int:
        """Get optimal FPS based on system performance - higher FPS"""
        if self.performance_mode == "performance":
            return 60  # High FPS for smooth processing
        elif self.performance_mode == "balanced":
            return 30
        else:  # efficiency
            return 20
    
    def should_use_gpu(self) -> bool:
        """Always use GPU unless critically overloaded"""
        return self.gpu_usage < 95 and self.gpu_memory_usage < 95
    
    def get_processing_threads(self) -> int:
        """Get optimal number of processing threads - use all available"""
        if self.performance_mode == "performance":
            return self.cpu_count  # Use ALL CPU cores
        elif self.performance_mode == "balanced":
            return max(self.cpu_count - 2, 4)  # Leave 2 cores for system
        else:  # efficiency
            return max(self.cpu_count // 2, 2)
    
    def record_frame_time(self, processing_time: float):
        """Record frame processing time for FPS calculation"""
        self.frame_processing_times.append(processing_time)
        
        # Keep only last 30 measurements
        if len(self.frame_processing_times) > 30:
            self.frame_processing_times = self.frame_processing_times[-30:]
        
        # Calculate average FPS
        if self.frame_processing_times:
            avg_time = sum(self.frame_processing_times) / len(self.frame_processing_times)
            self.average_fps = 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status"""
        return {
            "cpu": {
                "usage_percent": self.cpu_usage,
                "cores": self.cpu_count,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            "memory": {
                "usage_percent": self.memory_usage,
                "total_gb": self.total_memory / (1024**3),
                "available_gb": self.available_memory / (1024**3)
            },
            "gpu": {
                "usage_percent": self.gpu_usage,
                "memory_usage_percent": self.gpu_memory_usage
            },
            "disk": {
                "usage_percent": self.disk_usage
            },
            "performance": {
                "mode": self.performance_mode,
                "average_fps": self.average_fps,
                "target_fps": self.target_fps,
                "optimal_batch_size": self.get_optimal_batch_size(),
                "optimal_resolution": self.get_optimal_resolution(),
                "optimal_threads": self.get_processing_threads()
            },
            "recommendations": {
                "use_gpu": self.should_use_gpu(),
                "optimal_fps": self.get_optimal_fps()
            }
        }
    
    def print_status(self):
        """Print current resource status"""
        status = self.get_resource_status()
        
        print("\n" + "="*60)
        print("📊 DYNAMIC RESOURCE STATUS")
        print("="*60)
        print(f"🖥️  CPU: {status['cpu']['usage_percent']:.1f}% ({status['cpu']['cores']} cores)")
        print(f"💾 Memory: {status['memory']['usage_percent']:.1f}% ({status['memory']['available_gb']:.1f}GB available)")
        print(f"🎮 GPU: {status['gpu']['usage_percent']:.1f}% usage, {status['gpu']['memory_usage_percent']:.1f}% memory")
        print(f"💿 Disk: {status['disk']['usage_percent']:.1f}%")
        print(f"⚡ Mode: {status['performance']['mode'].upper()}")
        print(f"📈 FPS: {status['performance']['average_fps']:.1f} (target: {status['performance']['target_fps']})")
        print(f"🎯 Optimal: {status['performance']['optimal_resolution']} @ {status['recommendations']['optimal_fps']}fps")
        print(f"🔧 Threads: {status['performance']['optimal_threads']}, Batch: {status['performance']['optimal_batch_size']}")
        print(f"🎮 GPU Recommended: {'✅ Yes' if status['recommendations']['use_gpu'] else '❌ No'}")
        print("="*60)

# Global resource monitor instance
resource_monitor = ResourceMonitor()

def start_resource_monitoring():
    """Start global resource monitoring"""
    resource_monitor.start_monitoring()

def stop_resource_monitoring():
    """Stop global resource monitoring"""
    resource_monitor.stop_monitoring()

def get_dynamic_config() -> Dict[str, Any]:
    """Get dynamic configuration based on current resources"""
    return {
        "use_gpu": resource_monitor.should_use_gpu(),
        "batch_size": resource_monitor.get_optimal_batch_size(),
        "resolution": resource_monitor.get_optimal_resolution(),
        "fps": resource_monitor.get_optimal_fps(),
        "threads": resource_monitor.get_processing_threads(),
        "performance_mode": resource_monitor.performance_mode
    }
