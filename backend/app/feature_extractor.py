# app/feature_extractor.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_features(image):
    """Robust feature extraction with enhanced preprocessing"""
    try:
        # Handle different input types
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, np.ndarray):
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        else:
            raise ValueError("Input must be file path or numpy array")
        
        if img is None:
            logger.warning("Image is None")
            return None
            
        # Enhanced preprocessing
        img = cv2.resize(img, (640, 480))  # Standard size
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Initialize ORB detector with more features
        orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=10,
            WTA_K=2
        )
        
        # Detect and compute features
        kp, des = orb.detectAndCompute(img, None)
        
        if des is None or len(des) < 20:
            logger.warning(f"Insufficient features detected: {len(kp) if kp else 0} keypoints")
            return None
            
        logger.debug(f"Extracted {len(des)} features")
        return (kp, des)  # Return both keypoints and descriptors
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}", exc_info=True)
        return None