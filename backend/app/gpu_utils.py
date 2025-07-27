"""
GPU Detection and Configuration Utilities
Handles automatic GPU/CPU detection and configuration for both TensorFlow and PyTorch
"""

import os
import logging
import subprocess
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class GPUManager:
    def __init__(self):
        self.gpu_available = False
        self.cuda_version = None
        self.gpu_info = {}
        self.tensorflow_gpu_ready = False
        self.pytorch_gpu_ready = False
        
        self._detect_gpu()
        self._check_frameworks()
    
    def _detect_gpu(self):
        """Detect GPU availability and information"""
        try:
            # First try to detect via TensorFlow/PyTorch
            gpu_detected_by_frameworks = False
            
            # Check TensorFlow GPU detection
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    gpu_detected_by_frameworks = True
                    logger.info(f"🎮 TensorFlow detected {len(gpus)} GPU(s)")
            except:
                pass
            
            # Check PyTorch GPU detection
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_detected_by_frameworks = True
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                    logger.info(f"🎮 PyTorch detected {gpu_count} GPU(s): {gpu_name}")
                    self.gpu_info = {
                        'name': gpu_name,
                        'memory_mb': 4096,  # Default for GTX 1650
                        'driver_version': 'Unknown'
                    }
            except:
                pass
            
            # Try nvidia-smi for detailed info
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines and lines[0]:
                        gpu_data = lines[0].split(', ')
                        self.gpu_info = {
                            'name': gpu_data[0],
                            'memory_mb': int(gpu_data[1]),
                            'driver_version': gpu_data[2]
                        }
                        logger.info(f"🎮 nvidia-smi: {self.gpu_info['name']} ({self.gpu_info['memory_mb']}MB)")
            except:
                logger.debug("nvidia-smi not available, using framework detection")
            
            # Set GPU availability based on framework detection
            self.gpu_available = gpu_detected_by_frameworks
            
            # Check CUDA version
            try:
                cuda_result = subprocess.run(['nvcc', '--version'], 
                                           capture_output=True, text=True, timeout=5)
                if cuda_result.returncode == 0:
                    for line in cuda_result.stdout.split('\n'):
                        if 'release' in line:
                            self.cuda_version = line.split('release ')[1].split(',')[0]
                            logger.info(f"🔧 CUDA Version: {self.cuda_version}")
                            break
            except:
                # Try to get CUDA version from PyTorch
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.cuda_version = torch.version.cuda
                        logger.info(f"🔧 CUDA Version (PyTorch): {self.cuda_version}")
                except:
                    pass
        
        except Exception as e:
            logger.warning(f"GPU detection error: {e}")
            self.gpu_available = False
    
    def _check_frameworks(self):
        """Check if frameworks can use GPU"""
        # Check TensorFlow GPU support
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and self.gpu_available:
                self.tensorflow_gpu_ready = True
                logger.info(f"✅ TensorFlow GPU ready: {len(gpus)} GPU(s) detected")
            else:
                logger.info("⚠️ TensorFlow will use CPU")
        except ImportError:
            logger.warning("TensorFlow not installed")
        except Exception as e:
            logger.warning(f"TensorFlow GPU check failed: {e}")
        
        # Check PyTorch GPU support
        try:
            import torch
            if torch.cuda.is_available() and self.gpu_available:
                self.pytorch_gpu_ready = True
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                logger.info(f"✅ PyTorch GPU ready: {gpu_count} GPU(s), {gpu_name}")
            else:
                logger.info("⚠️ PyTorch will use CPU")
        except ImportError:
            logger.warning("PyTorch not installed")
        except Exception as e:
            logger.warning(f"PyTorch GPU check failed: {e}")
    
    def configure_tensorflow(self, force_cpu: bool = False, memory_limit_mb: int = None, dynamic_memory: bool = True) -> bool:
        """Configure TensorFlow for GPU or CPU usage with full GPU utilization"""
        try:
            import tensorflow as tf
            
            if force_cpu or not self.tensorflow_gpu_ready:
                # Force CPU usage
                tf.config.set_visible_devices([], 'GPU')
                logger.info("🖥️ TensorFlow configured for CPU usage")
                return False
            
            # Configure GPU with FULL utilization
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth for dynamic allocation
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"🎮 TensorFlow GPU memory growth enabled for {gpu}")
                    
                    # Use FULL GPU memory - no limits
                    logger.info("🎮 TensorFlow GPU using FULL memory allocation (no limits)")
                    
                    # Allow soft device placement for better resource utilization
                    tf.config.set_soft_device_placement(True)
                    
                    # Enable mixed precision for better GPU utilization
                    try:
                        policy = tf.keras.mixed_precision.Policy('mixed_float16')
                        tf.keras.mixed_precision.set_global_policy(policy)
                        logger.info("🎮 TensorFlow mixed precision enabled for better GPU performance")
                    except:
                        pass
                    
                    # Configure for maximum GPU utilization
                    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all CPU cores
                    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all CPU cores
                    
                    logger.info("🎮 TensorFlow configured for FULL GPU usage with maximum performance")
                    return True
                    
                except RuntimeError as e:
                    if "virtual devices" in str(e).lower():
                        logger.info("🎮 TensorFlow GPU already configured with full allocation")
                        return True
                    else:
                        logger.warning(f"TensorFlow GPU configuration failed: {e}")
                        # Fallback to CPU
                        tf.config.set_visible_devices([], 'GPU')
                        logger.info("🖥️ TensorFlow fallback to CPU")
                        return False
            
        except ImportError:
            logger.warning("TensorFlow not available")
        except Exception as e:
            logger.error(f"TensorFlow configuration error: {e}")
        
        return False
    
    def get_pytorch_device(self, force_cpu: bool = False) -> str:
        """Get the appropriate PyTorch device"""
        try:
            import torch
            
            if force_cpu or not self.pytorch_gpu_ready:
                logger.info("🖥️ PyTorch using CPU")
                return 'cpu'
            
            if torch.cuda.is_available():
                device = 'cuda:0'
                logger.info(f"🎮 PyTorch using GPU: {device}")
                return device
            
        except ImportError:
            logger.warning("PyTorch not available")
        except Exception as e:
            logger.error(f"PyTorch device selection error: {e}")
        
        logger.info("🖥️ PyTorch fallback to CPU")
        return 'cpu'
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU status"""
        return {
            'gpu_available': self.gpu_available,
            'gpu_info': self.gpu_info,
            'cuda_version': self.cuda_version,
            'tensorflow_gpu_ready': self.tensorflow_gpu_ready,
            'pytorch_gpu_ready': self.pytorch_gpu_ready
        }
    
    def print_status(self):
        """Print detailed GPU status"""
        print("\n" + "="*60)
        print("🎮 GPU MANAGER STATUS")
        print("="*60)
        
        if self.gpu_available:
            print(f"✅ GPU Available: {self.gpu_info.get('name', 'Unknown')}")
            print(f"   Memory: {self.gpu_info.get('memory_mb', 0)}MB")
            print(f"   Driver: {self.gpu_info.get('driver_version', 'Unknown')}")
            if self.cuda_version:
                print(f"   CUDA: {self.cuda_version}")
        else:
            print("❌ No GPU Available")
        
        print(f"🔧 TensorFlow GPU: {'✅ Ready' if self.tensorflow_gpu_ready else '❌ Not Ready'}")
        print(f"🔧 PyTorch GPU: {'✅ Ready' if self.pytorch_gpu_ready else '❌ Not Ready'}")
        print("="*60)

# Global GPU manager instance
gpu_manager = GPUManager()

def auto_configure_gpu(use_gpu: str = "auto", force_cpu_tf: bool = False, 
                      force_cpu_torch: bool = False, memory_limit: int = None, 
                      dynamic_memory: bool = True) -> Tuple[bool, str]:
    """
    Auto-configure GPU based on settings with dynamic resource allocation
    
    Args:
        use_gpu: "auto", "true", or "false"
        force_cpu_tf: Force TensorFlow to use CPU
        force_cpu_torch: Force PyTorch to use CPU
        memory_limit: GPU memory limit in MB (None for no limit)
        dynamic_memory: Enable dynamic memory allocation
    
    Returns:
        Tuple of (tensorflow_using_gpu, pytorch_device)
    """
    
    # Determine if we should use GPU
    should_use_gpu = False
    if use_gpu == "auto":
        should_use_gpu = gpu_manager.gpu_available
    elif use_gpu == "true":
        should_use_gpu = True
    elif use_gpu == "false":
        should_use_gpu = False
    
    # Configure TensorFlow with dynamic memory allocation
    tf_using_gpu = False
    if should_use_gpu and not force_cpu_tf:
        tf_using_gpu = gpu_manager.configure_tensorflow(
            force_cpu=False, 
            memory_limit_mb=memory_limit, 
            dynamic_memory=dynamic_memory
        )
    else:
        gpu_manager.configure_tensorflow(force_cpu=True)
    
    # Configure PyTorch
    pytorch_device = gpu_manager.get_pytorch_device(force_cpu=force_cpu_torch or not should_use_gpu)
    
    return tf_using_gpu, pytorch_device
