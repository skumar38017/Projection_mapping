"""
TensorFlow CUDA Error Fix
Handles CUDA_ERROR_INVALID_HANDLE by implementing proper fallback mechanisms
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def fix_tensorflow_cuda():
    """Fix TensorFlow CUDA issues by setting proper environment variables"""
    
    # Set TensorFlow environment variables to avoid CUDA issues
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Disable problematic CUDA operations
    os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP'] = '1'
    os.environ['TF_DISABLE_SPARSE_SOFTMAX_XENT'] = '1'
    
    # Use CPU for problematic operations
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU for TensorFlow
    
    logger.info("🔧 TensorFlow configured to use CPU (avoiding CUDA errors)")

def get_safe_tensorflow_model():
    """Get a TensorFlow model that works reliably"""
    try:
        # Fix CUDA issues first
        fix_tensorflow_cuda()
        
        import tensorflow as tf
        
        # Force CPU usage for TensorFlow
        tf.config.set_visible_devices([], 'GPU')
        
        # Use a simpler, more reliable model
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.models import Model
        
        # MobileNetV2 is more stable than EfficientNet
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        
        # Create feature extraction model
        model = Model(inputs=base_model.input, outputs=base_model.output)
        
        logger.info("✅ MobileNetV2 model loaded successfully (CPU)")
        return model
        
    except Exception as e:
        logger.error(f"TensorFlow model loading failed: {e}")
        return None

def extract_features_safe(model, image):
    """Safely extract features using TensorFlow model"""
    if model is None:
        return None
    
    try:
        import tensorflow as tf
        import cv2
        import numpy as np
        
        # Preprocess image
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
        
        if img is None:
            return None
        
        # Resize and preprocess
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Extract features
        with tf.device('/CPU:0'):  # Force CPU
            features = model.predict(img, verbose=0)
        
        return features.flatten()
        
    except Exception as e:
        logger.debug(f"Feature extraction failed: {e}")
        return None
