# app/object_detector.py
import cv2
import numpy as np
import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from app.config import settings
from app.utils import draw_bounding_box
from app.gpu_utils import auto_configure_gpu, gpu_manager
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self):
        # Configure GPU/CPU based on settings with dynamic allocation
        self.tf_using_gpu, self.pytorch_device = auto_configure_gpu(
            use_gpu=settings.USE_GPU,
            force_cpu_tf=settings.FORCE_CPU_TENSORFLOW,
            force_cpu_torch=settings.FORCE_CPU_PYTORCH,
            memory_limit=settings.GPU_MEMORY_LIMIT,
            dynamic_memory=settings.DYNAMIC_MEMORY
        )
        
        # Print GPU status
        gpu_manager.print_status()
        
        # Initialize multiple detection methods
        self.feature_extractor = self._init_feature_extractor()
        self.deep_feature_extractor = self._init_deep_feature_extractor()
        self.detection_model = self._init_detection_model()
        
        # Load reference features cache
        self.ref_features = []
        self.ref_deep_features = []
        self.ref_images_info = []
        self._load_reference_features()
        
        # Thresholds from settings
        self.feature_match_threshold = settings.FEATURE_MATCH_THRESHOLD
        self.deep_match_threshold = settings.DEEP_MATCH_THRESHOLD
        self.detection_confidence = settings.DETECTION_CONFIDENCE
        
    def _init_feature_extractor(self):
        """Initialize ORB feature extractor"""
        return cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=10,
            WTA_K=2
        )
        
    def _init_deep_feature_extractor(self):
        """Initialize deep feature extractor using EfficientNet"""
        try:
            # TensorFlow GPU/CPU configuration is already handled by gpu_manager
            base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
            model = Model(inputs=base_model.input, outputs=base_model.output)
            
            device_info = "GPU" if self.tf_using_gpu else "CPU"
            logger.info(f"✅ EfficientNetB0 model loaded successfully ({device_info})")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load EfficientNetB0: {e}")
            return None
        
    def _init_detection_model(self):
        """Initialize object detection model"""
        try:
            device = torch.device(self.pytorch_device)
            logger.info(f"🎯 Using {device} for object detection")
                
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            model.to(device)
            
            logger.info(f"✅ Faster R-CNN model loaded successfully ({device})")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load detection model: {e}")
            return None
    
    def _load_reference_features(self):
        """Pre-load all reference image features for faster matching"""
        logger.info("Loading reference features...")
        
        for filename in os.listdir(settings.ASSETS_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = settings.ASSETS_DIR / filename
                
                try:
                    # Load image
                    ref_img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    ref_color_img = cv2.imread(str(path))
                    
                    if ref_img is None or ref_color_img is None:
                        logger.warning(f"Failed to load reference: {filename}")
                        continue
                    
                    # Store image info
                    image_info = {
                        'filename': filename,
                        'path': str(path),
                        'size': ref_color_img.shape[:2],
                        'loaded_at': datetime.now()
                    }
                    self.ref_images_info.append(image_info)
                    
                    # Extract traditional features
                    features = self.extract_features(ref_img)
                    if features is not None:
                        self.ref_features.append((filename, features))
                    
                    # Extract deep features
                    if self.deep_feature_extractor is not None:
                        deep_features = self.extract_deep_features(ref_color_img)
                        if deep_features is not None:
                            self.ref_deep_features.append((filename, deep_features))
                    
                    logger.info(f"Loaded features for: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.ref_features)} traditional and {len(self.ref_deep_features)} deep feature sets")
        
    def extract_features(self, image):
        """Extract features using ORB"""
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
                
            # Preprocessing
            img = cv2.resize(img, (640, 480))
            img = cv2.equalizeHist(img)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Feature extraction
            kp, des = self.feature_extractor.detectAndCompute(img, None)
            
            if des is None or len(des) < 20:
                logger.debug(f"Insufficient features detected: {len(kp) if kp else 0} keypoints")
                return None
                
            logger.debug(f"Extracted {len(des)} features")
            return (kp, des)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return None
            
    def extract_deep_features(self, image):
        """Extract deep features using EfficientNet"""
        try:
            if self.deep_feature_extractor is None:
                return None
                
            if isinstance(image, str):
                img = cv2.imread(image)
            elif isinstance(image, np.ndarray):
                img = image
            else:
                raise ValueError("Input must be file path or numpy array")
                
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            
            features = self.deep_feature_extractor.predict(img, verbose=0)
            return features.flatten()
            
        except Exception as e:
            logger.error(f"Deep feature extraction failed: {str(e)}")
            return None
            
    def detect_objects(self, image):
        """Detect objects using Faster R-CNN"""
        try:
            if self.detection_model is None:
                return None, None, None
                
            if isinstance(image, str):
                img = cv2.imread(image)
            elif isinstance(image, np.ndarray):
                img = image
            else:
                raise ValueError("Input must be file path or numpy array")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(img).unsqueeze(0)
            
            # Move tensor to the appropriate device
            device = torch.device(self.pytorch_device)
            img_tensor = img_tensor.to(device)
                
            with torch.no_grad():
                predictions = self.detection_model(img_tensor)
                
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            
            # Filter predictions by confidence
            valid_detections = scores > self.detection_confidence
            boxes = boxes[valid_detections]
            scores = scores[valid_detections]
            labels = labels[valid_detections]
            
            return boxes, scores, labels
            
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            return None, None, None
            
    def match_features(self, frame_kp_des, ref_kp_des):
        """Robust feature matching for ORB descriptors"""
        try:
            if frame_kp_des is None or ref_kp_des is None:
                return 0.0

            frame_kp, frame_des = frame_kp_des
            ref_kp, ref_des = ref_kp_des

            # ORB descriptors are binary - use Hamming distance
            if frame_des.dtype != np.uint8:
                frame_des = frame_des.astype(np.uint8)
            if ref_des.dtype != np.uint8:
                ref_des = ref_des.astype(np.uint8)

            # Create BFMatcher with Hamming distance
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # Match descriptors
            matches = bf.match(frame_des, ref_des)
            
            if len(matches) == 0:
                return 0.0
                
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate score based on good matches
            good_matches = [m for m in matches if m.distance < 50]
            
            if len(good_matches) < 15:
                return 0.0
                
            # Calculate homography for verification
            if len(good_matches) >= 4:
                src_pts = np.float32([frame_kp[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mask is not None:
                    inlier_ratio = np.sum(mask) / len(mask)
                    match_score = len(good_matches) / len(matches)
                    final_score = (0.6 * inlier_ratio) + (0.4 * match_score)
                    return final_score
            
            # Fallback score
            return len(good_matches) / min(len(frame_des), len(ref_des))
            
        except Exception as e:
            logger.error(f"Matching failed: {str(e)}")
            return 0.0
            
    def match_deep_features(self, frame_features, ref_features_list):
        """Match features using deep learning embeddings"""
        try:
            if frame_features is None or len(ref_features_list) == 0:
                return None, 0.0
                
            # Calculate cosine similarity with all reference features
            best_match = None
            best_score = 0.0
            
            for filename, ref_features in ref_features_list:
                # Normalize vectors
                frame_norm = frame_features / np.linalg.norm(frame_features)
                ref_norm = ref_features / np.linalg.norm(ref_features)
                
                # Calculate cosine similarity
                similarity = np.dot(frame_norm, ref_norm)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = filename
            
            return best_match, best_score
            
        except Exception as e:
            logger.error(f"Deep matching failed: {str(e)}")
            return None, 0.0

    def get_reference_images_info(self):
        """Get information about loaded reference images"""
        return self.ref_images_info

def verify_object_from_frame(frame: np.ndarray):
    """Enhanced object verification with multiple methods"""
    start_time = time.time()
    
    try:
        # Initialize detector (singleton pattern could be used for optimization)
        detector = ObjectDetector()
        
        # Save debug frame
        debug_path = settings.DEBUG_DIR / f'frame_{int(time.time())}.jpg'
        cv2.imwrite(str(debug_path), frame)
        
        # Extract features using multiple methods
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frame_features = detector.extract_features(frame_gray)
        frame_deep_features = detector.extract_deep_features(frame)
        
        processed_frame = frame.copy()
        best_match = None
        best_score = 0
        best_method = None
        match_details = {}
        
        # 1. Traditional feature matching
        if frame_features is not None and len(detector.ref_features) > 0:
            for filename, ref_features in detector.ref_features:
                score = detector.match_features(frame_features, ref_features)
                if score > best_score:
                    best_score = score
                    best_match = filename
                    best_method = 'feature'
                    match_details = {
                        'method': 'ORB Feature Matching',
                        'score': float(score),  # Convert to Python float
                        'threshold': float(detector.feature_match_threshold)  # Convert to Python float
                    }
                
        # 2. Deep feature matching
        if frame_deep_features is not None and len(detector.ref_deep_features) > 0:
            match, score = detector.match_deep_features(frame_deep_features, detector.ref_deep_features)
            if match is not None and score > best_score:
                best_score = score
                best_match = match
                best_method = 'deep'
                match_details = {
                    'method': 'Deep Learning (EfficientNet)',
                    'score': float(score),  # Convert to Python float
                    'threshold': float(detector.deep_match_threshold)  # Convert to Python float
                }
        
        # 3. Object detection (optional enhancement)
        boxes, scores, labels = detector.detect_objects(frame)
        if boxes is not None and len(boxes) > 0:
            detection_score = np.max(scores)
            if detection_score > detector.detection_confidence:
                match_details['detected_objects'] = int(len(boxes))  # Convert to Python int
                match_details['detection_confidence'] = float(detection_score)  # Convert to Python float
        
        # Determine if match is successful
        success = False
        if best_method == 'feature' and best_score > detector.feature_match_threshold:
            success = True
        elif best_method == 'deep' and best_score > detector.deep_match_threshold:
            success = True
        
        processing_time = time.time() - start_time
        
        # Draw results on frame
        if success:
            label = f"{best_match} ({best_score:.3f})"
            processed_frame = draw_bounding_box(processed_frame, label, color=(0, 255, 0))
        else:
            processed_frame = draw_bounding_box(processed_frame, "No Match", color=(0, 0, 255))
        
        # Console output
        if success:
            print(f"✅ MATCH FOUND: {best_match} | Score: {best_score:.3f} | Method: {best_method} | Time: {processing_time:.3f}s")
        else:
            print(f"❌ NO MATCH | Best Score: {best_score:.3f} | Time: {processing_time:.3f}s")
        
        return {
            "success": success,
            "match_path": best_match,
            "score": float(best_score),  # Convert numpy float to Python float
            "method": best_method,
            "details": match_details,
            "processing_time": float(processing_time),  # Convert numpy float to Python float
            "timestamp": datetime.now().isoformat()
        }, processed_frame
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        print(f"❌ ERROR: {str(e)}")
        return {
            "success": False,
            "match_path": None,
            "score": 0.0,
            "method": None,
            "details": {"error": str(e)},
            "processing_time": float(time.time() - start_time),  # Convert to Python float
            "timestamp": datetime.now().isoformat()
        }, frame