"""
Real-time Object Detection with MediaPipe, YOLO, and optimized processing
Ultra-fast, low-latency detection for real-time streaming
"""

import cv2
import numpy as np
import time
import logging
from typing import Tuple, Dict, Any, Optional, List
import threading
from datetime import datetime
import os

# High-performance imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from app.config import settings
from app.utils import draw_bounding_box, make_json_serializable
from app.resource_monitor import resource_monitor
from app.tensorflow_fix import get_safe_tensorflow_model, extract_features_safe

logger = logging.getLogger(__name__)

class RealtimeDetector:
    def __init__(self):
        self.detection_methods = []
        self.reference_features = {}
        self.processing_times = []
        
        # Initialize available detection methods
        self._init_mediapipe()
        self._init_yolo()
        self._init_opencv_optimized()
        self._init_tensorflow_safe()
        
        # Load reference images
        self._load_reference_images()
        
        # Performance settings
        self.target_fps = 30
        self.max_processing_time = 0.033  # 33ms for 30fps
        self.frame_skip = 1  # Process every frame initially
        
        logger.info(f"🚀 RealtimeDetector initialized with {len(self.detection_methods)} methods")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe for ultra-fast detection"""
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe not available")
            return
        
        try:
            # MediaPipe Object Detection
            self.mp_objectron = mp.solutions.objectron
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Initialize Objectron for 3D object detection
            self.objectron = self.mp_objectron.Objectron(
                static_image_mode=False,
                max_num_objects=5,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_name='Cup'  # Can detect cup-like objects (watches, etc.)
            )
            
            # MediaPipe Hands for gesture-based detection
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.detection_methods.append('mediapipe')
            logger.info("✅ MediaPipe initialized for real-time detection")
            
        except Exception as e:
            logger.error(f"MediaPipe initialization failed: {e}")
    
    def _init_yolo(self):
        """Initialize YOLO for fast object detection"""
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available")
            return
        
        try:
            # Use YOLOv8 nano for maximum speed
            self.yolo_model = YOLO('yolov8n.pt')  # Nano version for speed
            
            # Configure for GPU if available
            device = 'cuda:0' if resource_monitor.should_use_gpu() else 'cpu'
            self.yolo_model.to(device)
            
            # Optimize for inference
            self.yolo_model.fuse()  # Fuse layers for speed
            
            self.detection_methods.append('yolo')
            logger.info(f"✅ YOLOv8 initialized on {device}")
            
        except Exception as e:
            logger.error(f"YOLO initialization failed: {e}")
    
    def _init_opencv_optimized(self):
        """Initialize optimized OpenCV detection"""
        try:
            # Optimized ORB detector
            self.orb = cv2.ORB_create(
                nfeatures=500,  # Reduced for speed
                scaleFactor=1.2,
                nlevels=4,  # Reduced levels
                edgeThreshold=15,
                patchSize=31,
                fastThreshold=20  # Higher threshold for speed
            )
            
            # SIFT detector for high-quality features
            try:
                self.sift = cv2.SIFT_create(nfeatures=300)  # Reduced features
            except:
                self.sift = None
            
            # Template matching setup
            self.template_matcher = cv2.TM_CCOEFF_NORMED
            
            self.detection_methods.append('opencv_optimized')
            logger.info("✅ Optimized OpenCV detection initialized")
            
        except Exception as e:
            logger.error(f"OpenCV optimization failed: {e}")
    
    def _init_tensorflow_safe(self):
        """Initialize safe TensorFlow model for feature extraction"""
        try:
            self.tf_model = get_safe_tensorflow_model()
            if self.tf_model:
                self.detection_methods.append('tensorflow_safe')
                logger.info("✅ Safe TensorFlow model initialized")
        except Exception as e:
            logger.error(f"Safe TensorFlow initialization failed: {e}")
            self.tf_model = None
    
    def _load_reference_images(self):
        """Load and preprocess reference images for fast matching"""
        try:
            assets_dir = settings.ASSETS_DIR
            if not assets_dir.exists():
                logger.warning(f"Assets directory not found: {assets_dir}")
                return
            
            for filename in os.listdir(assets_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = assets_dir / filename
                    
                    # Load image
                    img = cv2.imread(str(filepath))
                    if img is None:
                        continue
                    
                    # Preprocess for different scales
                    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
                    templates = {}
                    
                    for scale in scales:
                        h, w = img.shape[:2]
                        new_h, new_w = int(h * scale), int(w * scale)
                        if new_h > 0 and new_w > 0:
                            scaled = cv2.resize(img, (new_w, new_h))
                            templates[scale] = scaled
                    
                    # Extract features
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # ORB features
                    kp_orb, des_orb = self.orb.detectAndCompute(gray, None)
                    
                    # SIFT features (if available)
                    kp_sift, des_sift = None, None
                    if self.sift:
                        kp_sift, des_sift = self.sift.detectAndCompute(gray, None)
                    
                    self.reference_features[filename] = {
                        'templates': templates,
                        'orb': (kp_orb, des_orb),
                        'sift': (kp_sift, des_sift),
                        'original': img,
                        'gray': gray
                    }
                    
                    logger.info(f"📷 Loaded reference: {filename}")
            
            logger.info(f"✅ Loaded {len(self.reference_features)} reference images")
            
        except Exception as e:
            logger.error(f"Reference loading failed: {e}")
    
    def detect_fast(self, frame: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """Ultra-fast detection with multiple methods"""
        start_time = time.time()
        
        # Adaptive frame skipping based on performance
        if len(self.processing_times) > 10:
            avg_time = sum(self.processing_times[-10:]) / 10
            if avg_time > self.max_processing_time:
                self.frame_skip = min(self.frame_skip + 1, 3)
            else:
                self.frame_skip = max(self.frame_skip - 1, 1)
        
        # Skip frames if needed
        if hasattr(self, '_frame_counter'):
            self._frame_counter += 1
        else:
            self._frame_counter = 0
        
        if self._frame_counter % self.frame_skip != 0:
            return self._get_no_match_result(start_time), frame
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame_small = cv2.resize(frame, (new_width, new_height))
        else:
            frame_small = frame.copy()
            scale = 1.0
        
        # Try detection methods in order of speed
        result = None
        processed_frame = frame.copy()
        
        # 1. Fast template matching
        if 'opencv_optimized' in self.detection_methods:
            result = self._detect_template_matching(frame_small, scale)
            if result and result['success']:
                processed_frame = self._draw_detection_result(frame, result, scale)
        
        # 2. MediaPipe detection (if template matching failed)
        if not (result and result['success']) and 'mediapipe' in self.detection_methods:
            mp_result = self._detect_mediapipe(frame_small)
            if mp_result and mp_result['success']:
                result = mp_result
                processed_frame = self._draw_detection_result(frame, result, scale)
        
        # 3. YOLO detection (if others failed and we have time)
        processing_time = time.time() - start_time
        if not (result and result['success']) and 'yolo' in self.detection_methods and processing_time < 0.02:
            yolo_result = self._detect_yolo(frame_small)
            if yolo_result and yolo_result['success']:
                result = yolo_result
                processed_frame = self._draw_detection_result(frame, result, scale)
        
        # Fallback result
        if not result:
            result = self._get_no_match_result(start_time)
        
        # Update performance metrics
        total_time = time.time() - start_time
        self.processing_times.append(total_time)
        if len(self.processing_times) > 30:
            self.processing_times = self.processing_times[-30:]
        
        result['processing_time'] = total_time
        result['fps'] = 1.0 / total_time if total_time > 0 else 0
        
        return make_json_serializable(result), processed_frame
    
    def _detect_template_matching(self, frame: np.ndarray, scale: float = 1.0) -> Optional[Dict[str, Any]]:
        """Ultra-fast template matching"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            best_match = None
            best_score = 0
            best_location = None
            
            for ref_name, ref_data in self.reference_features.items():
                templates = ref_data['templates']
                
                # Try different scales
                for template_scale, template in templates.items():
                    if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
                        continue
                    
                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    
                    # Template matching
                    result = cv2.matchTemplate(gray, template_gray, self.template_matcher)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_score:
                        best_score = max_val
                        best_match = ref_name
                        best_location = max_loc
            
            # Check if match is good enough
            if best_score > 0.6:  # Lower threshold for speed
                return {
                    'success': True,
                    'match_path': best_match,
                    'score': float(best_score),
                    'method': 'template_matching',
                    'location': best_location,
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.debug(f"Template matching error: {e}")
        
        return None
    
    def _detect_mediapipe(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """MediaPipe-based detection"""
        if not MEDIAPIPE_AVAILABLE:
            return None
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Object detection with Objectron
            results = self.objectron.process(rgb_frame)
            
            if results.detected_objects:
                # Found objects
                confidence = 0.8  # MediaPipe confidence
                return {
                    'success': True,
                    'match_path': 'mediapipe_object',
                    'score': confidence,
                    'method': 'mediapipe',
                    'objects_count': len(results.detected_objects),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Try hand detection as fallback
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                return {
                    'success': True,
                    'match_path': 'hand_gesture',
                    'score': 0.7,
                    'method': 'mediapipe_hands',
                    'hands_count': len(hand_results.multi_hand_landmarks),
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.debug(f"MediaPipe detection error: {e}")
        
        return None
    
    def _detect_yolo(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """YOLO-based detection"""
        if not YOLO_AVAILABLE:
            return None
        
        try:
            # Run YOLO inference
            results = self.yolo_model(frame, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                # Check for relevant objects (watches, clocks, etc.)
                relevant_classes = [74, 84]  # clock, book (COCO classes)
                
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:  # Any object with good confidence
                            return {
                                'success': True,
                                'match_path': f'yolo_class_{class_id}',
                                'score': confidence,
                                'method': 'yolo',
                                'class_id': class_id,
                                'bbox': box.xyxy[0].tolist(),
                                'timestamp': datetime.now().isoformat()
                            }
            
        except Exception as e:
            logger.debug(f"YOLO detection error: {e}")
        
        return None
    
    def _draw_detection_result(self, frame: np.ndarray, result: Dict[str, Any], scale: float = 1.0) -> np.ndarray:
        """Draw detection results on frame"""
        try:
            processed = frame.copy()
            
            if result['success']:
                # Draw bounding box and label
                label = f"{result.get('match_path', 'Object')} ({result['score']:.2f})"
                color = (0, 255, 0)  # Green for success
                
                # Draw based on detection method
                if 'location' in result:
                    # Template matching - draw rectangle
                    x, y = result['location']
                    x, y = int(x / scale), int(y / scale)
                    w, h = 100, 100  # Approximate size
                    cv2.rectangle(processed, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(processed, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                elif 'bbox' in result:
                    # YOLO - draw actual bounding box
                    bbox = result['bbox']
                    x1, y1, x2, y2 = [int(coord / scale) for coord in bbox]
                    cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(processed, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                else:
                    # General detection - draw center indicator
                    h, w = processed.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    cv2.circle(processed, (center_x, center_y), 50, color, 3)
                    cv2.putText(processed, label, (center_x - 100, center_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            else:
                # No match - draw red indicator
                processed = draw_bounding_box(processed, "No Match", color=(0, 0, 255))
            
            return processed
            
        except Exception as e:
            logger.debug(f"Drawing error: {e}")
            return frame
    
    def _get_no_match_result(self, start_time: float) -> Dict[str, Any]:
        """Get no match result"""
        return {
            'success': False,
            'match_path': None,
            'score': 0.0,
            'method': 'realtime_detector',
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.processing_times:
            return {'avg_fps': 0, 'avg_processing_time': 0}
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_fps': avg_fps,
            'avg_processing_time': avg_time,
            'frame_skip': self.frame_skip,
            'detection_methods': self.detection_methods,
            'target_fps': self.target_fps
        }

# Global realtime detector instance
realtime_detector = None

def get_realtime_detector() -> RealtimeDetector:
    """Get or create realtime detector instance"""
    global realtime_detector
    if realtime_detector is None:
        realtime_detector = RealtimeDetector()
    return realtime_detector

def verify_object_realtime(frame: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
    """Real-time object verification - main entry point"""
    detector = get_realtime_detector()
    return detector.detect_fast(frame)
