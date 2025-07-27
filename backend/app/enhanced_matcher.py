"""
Enhanced Object Matcher - Combines traditional feature matching with deep learning embeddings
Uses both ORB/SIFT features and CNN embeddings (CLIP, ResNet, MobileNet) for robust matching
Integrates with Pinecone and FAISS for vector similarity search
60% threshold: >=60% = "Matched", <60% = "Not Matched"
"""

import cv2
import numpy as np
import time
import logging
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

from app.config import settings
from app.utils import make_json_serializable
from app.network_broadcaster import get_network_broadcaster
from app.embedding_extractor import get_embedding_extractor
from app.vector_search import get_vector_search_engine

logger = logging.getLogger(__name__)

class EnhancedObjectMatcher:
    """
    Enhanced object matcher combining traditional computer vision with deep learning
    """
    
    def __init__(self):
        # Traditional feature matching
        self.stored_images = {}
        self.stored_features = {}
        self.matcher = cv2.BFMatcher()
        
        # Enhanced feature detector for better matching
        self.detector = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=10,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10
        )
        
        # SIFT detector for high-quality features
        try:
            self.sift_detector = cv2.SIFT_create(nfeatures=1500)
            logger.info("✅ SIFT detector initialized")
        except:
            self.sift_detector = None
            logger.warning("⚠️ SIFT detector not available")
        
        # Deep learning components
        self.embedding_extractor = get_embedding_extractor()
        self.vector_search_engine = get_vector_search_engine()
        
        # Network broadcaster
        self.broadcaster = get_network_broadcaster()
        
        # Load stored images and extract features
        self._load_stored_images()
        
        logger.info(f"✅ Enhanced matcher initialized with {len(self.stored_images)} stored images")
    
    def _load_stored_images(self):
        """Load stored images and extract both traditional and deep features"""
        try:
            assets_dir = settings.ASSETS_DIR
            if not assets_dir.exists():
                logger.warning(f"Assets directory not found: {assets_dir}")
                return
            
            print(f"\n🧠 [ENHANCED_MATCHING] Loading reference images from {assets_dir}")
            
            for filename in assets_dir.iterdir():
                if filename.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    self._process_reference_image(filename.name)
            
            logger.info(f"✅ Loaded {len(self.stored_images)} stored images with enhanced features")
            
        except Exception as e:
            logger.error(f"Error loading stored images: {e}")
    
    def _process_reference_image(self, filename: str):
        """Process a single reference image with both traditional and deep features"""
        filepath = settings.ASSETS_DIR / filename
        
        try:
            image = cv2.imread(str(filepath))
            if image is None:
                logger.warning(f"Could not load image: {filename}")
                return
            
            print(f"   📸 [PROCESSING] {filename}")
            
            # Convert to grayscale for traditional features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract traditional features (ORB)
            orb_keypoints, orb_descriptors = self.detector.detectAndCompute(gray, None)
            
            # Extract SIFT features if available
            sift_keypoints, sift_descriptors = None, None
            if self.sift_detector:
                sift_keypoints, sift_descriptors = self.sift_detector.detectAndCompute(gray, None)
            
            # Extract deep learning embeddings
            embeddings = self.embedding_extractor.extract_multiple_embeddings(image)
            
            # Store all features
            self.stored_images[filename] = {
                'image': image,
                'gray': gray,
                'orb_keypoints': orb_keypoints,
                'orb_descriptors': orb_descriptors,
                'sift_keypoints': sift_keypoints,
                'sift_descriptors': sift_descriptors,
                'embeddings': embeddings,
                'path': str(filepath)
            }
            
            orb_count = len(orb_descriptors) if orb_descriptors is not None else 0
            sift_count = len(sift_descriptors) if sift_descriptors is not None else 0
            embedding_count = len(embeddings)
            
            print(f"   ✅ [FEATURES] ORB: {orb_count}, SIFT: {sift_count}, Embeddings: {embedding_count}")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
    
    def match_with_stored_images(self, camera_frame: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Enhanced matching using both traditional features and deep learning embeddings
        60% threshold for final decision
        """
        start_time = time.time()
        
        print(f"\n🔍 [ENHANCED_MATCHING] Processing frame at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        try:
            # Convert camera frame to grayscale for traditional features
            if len(camera_frame.shape) == 3:
                frame_gray = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = camera_frame
            
            # Method 1: Traditional feature matching (ORB + SIFT)
            traditional_result = self._traditional_matching(frame_gray, camera_frame)
            
            # Method 2: Deep learning vector search
            vector_result = self._vector_search_matching(camera_frame)
            
            # Method 3: Combine results with weighted scoring
            final_result = self._combine_matching_results(traditional_result, vector_result, start_time)
            
            # Apply 60% threshold decision
            final_decision = self._apply_threshold_decision(final_result)
            
            # Draw result on frame
            processed_frame = self._draw_enhanced_result(camera_frame, final_decision)
            
            # Broadcast to network
            self._broadcast_enhanced_result(final_decision, camera_frame, start_time)
            
            return make_json_serializable(final_decision), processed_frame
            
        except Exception as e:
            print(f"💥 [ERROR] Enhanced matching failed: {e}")
            logger.error(f"Enhanced matching error: {e}")
            
            error_result = {
                'success': False,
                'match_result': 'Not Matched',
                'match_found': False,
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }
            
            return make_json_serializable(error_result), camera_frame
    
    def _traditional_matching(self, frame_gray: np.ndarray, camera_frame: np.ndarray) -> Dict[str, Any]:
        """Traditional ORB + SIFT feature matching"""
        try:
            # Extract features from camera frame
            orb_keypoints, orb_descriptors = self.detector.detectAndCompute(frame_gray, None)
            sift_keypoints, sift_descriptors = None, None
            
            if self.sift_detector:
                sift_keypoints, sift_descriptors = self.sift_detector.detectAndCompute(frame_gray, None)
            
            orb_count = len(orb_descriptors) if orb_descriptors is not None else 0
            sift_count = len(sift_descriptors) if sift_descriptors is not None else 0
            
            print(f"   🔍 [TRADITIONAL] Extracted ORB={orb_count}, SIFT={sift_count} features")
            
            if orb_count < 10 and sift_count < 10:
                return {'method': 'traditional', 'score': 0.0, 'matched_image': None, 'confidence': 0.0}
            
            best_match = None
            best_score = 0.0
            best_method = ""
            
            # Compare with each stored image
            for filename, stored_data in self.stored_images.items():
                # ORB matching
                orb_score, orb_matches = self._match_orb_features(
                    orb_descriptors, stored_data['orb_descriptors']
                )
                
                # SIFT matching
                sift_score, sift_matches = 0, 0
                if sift_descriptors is not None and stored_data['sift_descriptors'] is not None:
                    sift_score, sift_matches = self._match_sift_features(
                        sift_descriptors, stored_data['sift_descriptors']
                    )
                
                # Use the best score
                if orb_score > sift_score:
                    current_score = orb_score
                    current_method = "ORB"
                else:
                    current_score = sift_score
                    current_method = "SIFT"
                
                if current_score > best_score:
                    best_score = current_score
                    best_match = filename
                    best_method = current_method
            
            return {
                'method': 'traditional',
                'score': best_score,
                'matched_image': best_match,
                'confidence': best_score * 100,
                'algorithm': best_method
            }
            
        except Exception as e:
            logger.error(f"Traditional matching error: {e}")
            return {'method': 'traditional', 'score': 0.0, 'matched_image': None, 'confidence': 0.0}
    
    def _vector_search_matching(self, camera_frame: np.ndarray) -> Dict[str, Any]:
        """Deep learning vector search matching"""
        try:
            # Use vector search engine for similarity search
            best_match, similarity_score = self.vector_search_engine.get_best_match(
                camera_frame, threshold=0.0  # We'll apply our own threshold later
            )
            
            print(f"   🧠 [VECTOR_SEARCH] Best match: {best_match}, Score: {similarity_score:.3f}")
            
            return {
                'method': 'vector_search',
                'score': similarity_score,
                'matched_image': best_match,
                'confidence': similarity_score * 100,
                'algorithm': self.embedding_extractor.get_best_embedding_model()
            }
            
        except Exception as e:
            logger.error(f"Vector search matching error: {e}")
            return {'method': 'vector_search', 'score': 0.0, 'matched_image': None, 'confidence': 0.0}
    
    def _combine_matching_results(self, traditional: Dict[str, Any], vector: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Combine traditional and vector search results with weighted scoring"""
        
        # Weights for different methods (can be tuned)
        traditional_weight = 0.3  # 30% weight for traditional features
        vector_weight = 0.7       # 70% weight for deep learning
        
        # Calculate weighted scores
        traditional_score = traditional['score'] * traditional_weight
        vector_score = vector['score'] * vector_weight
        
        # Combined score
        combined_score = traditional_score + vector_score
        
        # Determine best match (prefer vector search if scores are close)
        if vector['matched_image'] and vector['score'] > 0.1:
            best_match = vector['matched_image']
            primary_method = f"Vector({vector['algorithm']})"
        elif traditional['matched_image'] and traditional['score'] > 0.1:
            best_match = traditional['matched_image']
            primary_method = f"Traditional({traditional['algorithm']})"
        else:
            best_match = None
            primary_method = "None"
        
        processing_time = time.time() - start_time
        
        print(f"   ⚖️ [COMBINED] Traditional: {traditional_score:.3f}, Vector: {vector_score:.3f}")
        print(f"   🎯 [RESULT] Combined score: {combined_score:.3f} ({combined_score*100:.1f}%)")
        print(f"   📸 [MATCH] {best_match} via {primary_method}")
        print(f"   ⏱️ [TIME] {processing_time*1000:.1f}ms")
        
        return {
            'combined_score': combined_score,
            'combined_confidence': combined_score * 100,
            'matched_image': best_match,
            'primary_method': primary_method,
            'traditional_result': traditional,
            'vector_result': vector,
            'processing_time_ms': processing_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
    
    def _apply_threshold_decision(self, combined_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply 60% threshold decision to combined results"""
        
        THRESHOLD = 60.0  # 60% threshold
        confidence = combined_result['combined_confidence']
        
        # Decision logic
        if confidence >= THRESHOLD:
            match_result = "Matched"
            match_found = True
            threshold_met = True
            print(f"   ✅ [DECISION] MATCHED ({confidence:.1f}% ≥ {THRESHOLD}%)")
        else:
            match_result = "Not Matched"
            match_found = False
            threshold_met = False
            print(f"   ❌ [DECISION] NOT MATCHED ({confidence:.1f}% < {THRESHOLD}%)")
        
        return {
            'success': True,
            'match_result': match_result,
            'match_found': match_found,
            'matched_image': combined_result['matched_image'] if match_found else None,
            'similarity_score': combined_result['combined_score'],
            'confidence': confidence,
            'threshold_met': threshold_met,
            'processing_time_ms': combined_result['processing_time_ms'],
            'timestamp': combined_result['timestamp'],
            'method': combined_result['primary_method'],
            'details': {
                'decision_threshold': THRESHOLD,
                'traditional_score': combined_result['traditional_result']['confidence'],
                'vector_score': combined_result['vector_result']['confidence'],
                'combined_algorithm': combined_result['primary_method'],
                'reference_image': combined_result['matched_image']
            }
        }
    
    def _match_orb_features(self, frame_descriptors, stored_descriptors):
        """Match ORB features with improved scoring"""
        if frame_descriptors is None or stored_descriptors is None:
            return 0, 0
        
        try:
            matches = self.matcher.knnMatch(frame_descriptors, stored_descriptors, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) >= 10:
                min_descriptors = min(len(frame_descriptors), len(stored_descriptors))
                raw_score = len(good_matches) / min_descriptors
                scaled_score = min(raw_score * 1.5, 1.0)
                return scaled_score, len(good_matches)
            
            return 0, 0
        except Exception as e:
            logger.debug(f"ORB matching error: {e}")
            return 0, 0
    
    def _match_sift_features(self, frame_descriptors, stored_descriptors):
        """Match SIFT features with improved scoring"""
        if frame_descriptors is None or stored_descriptors is None:
            return 0, 0
        
        try:
            matches = self.matcher.knnMatch(frame_descriptors, stored_descriptors, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) >= 8:
                min_descriptors = min(len(frame_descriptors), len(stored_descriptors))
                raw_score = len(good_matches) / min_descriptors
                scaled_score = min(raw_score * 1.8, 1.0)
                return scaled_score, len(good_matches)
            
            return 0, 0
        except Exception as e:
            logger.debug(f"SIFT matching error: {e}")
            return 0, 0
    
    def _draw_enhanced_result(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw enhanced matching result on frame"""
        result_frame = frame.copy()
        h, w = result_frame.shape[:2]
        
        # Choose colors based on match result
        if result['match_result'] == "Matched":
            border_color = (0, 255, 0)  # Green
            bg_color = (0, 255, 0)
            text_color = (0, 0, 0)
            status_symbol = "✅"
        else:
            border_color = (0, 0, 255)  # Red
            bg_color = (0, 0, 255)
            text_color = (255, 255, 255)
            status_symbol = "❌"
        
        # Draw border
        cv2.rectangle(result_frame, (10, 10), (w-10, h-10), border_color, 4)
        
        # Prepare text lines
        lines = [
            f"{status_symbol} {result['match_result'].upper()}",
            f"Confidence: {result['confidence']:.1f}% (60% threshold)",
            f"Reference: {result['matched_image'] if result['matched_image'] else 'None'}",
            f"Method: {result['method']}",
            f"Time: {result['processing_time_ms']:.1f}ms"
        ]
        
        # Calculate text background size
        line_height = 25
        text_height = len(lines) * line_height + 20
        text_width = 550
        
        # Draw text background
        cv2.rectangle(result_frame, (15, 15), (15 + text_width, 15 + text_height), bg_color, -1)
        
        # Draw text lines
        for i, line in enumerate(lines):
            y_pos = 40 + (i * line_height)
            font_size = 0.7 if i == 0 else 0.6
            thickness = 2 if i == 0 else 1
            cv2.putText(result_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, thickness)
        
        # Draw confidence bar
        bar_x = 20
        bar_y = h - 60
        bar_width = 400
        bar_height = 20
        
        # Background bar
        cv2.rectangle(result_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (128, 128, 128), -1)
        
        # Threshold line (60%)
        threshold_x = bar_x + int(bar_width * 0.6)
        cv2.line(result_frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_height + 5), (255, 255, 255), 2)
        
        # Confidence bar
        if result['confidence'] > 0:
            confidence_width = int(bar_width * min(result['confidence'] / 100, 1.0))
            confidence_color = (0, 255, 0) if result['confidence'] >= 60 else (0, 0, 255)
            cv2.rectangle(result_frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), confidence_color, -1)
        
        # Threshold label
        cv2.putText(result_frame, "60%", (threshold_x - 15, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def _broadcast_enhanced_result(self, result: Dict[str, Any], frame: np.ndarray, start_time: float):
        """Broadcast enhanced matching result to network"""
        try:
            broadcast_data = {
                'timestamp': result['timestamp'],
                'detection': {
                    'object_detected': True,
                    'match_found': result['match_found'],
                    'match_result': result['match_result'],
                    'confidence': result['confidence'],
                    'reference_image': result.get('matched_image', None),
                    'similarity_score': result['similarity_score'],
                    'method': result['method']
                },
                'performance': {
                    'processing_time_ms': result['processing_time_ms'],
                    'frame_shape': list(frame.shape),
                    'algorithm': result['method']
                },
                'system': {
                    'threshold': 60.0,
                    'threshold_met': result['threshold_met'],
                    'matching_type': 'enhanced_hybrid'
                }
            }
            
            self.broadcaster.broadcast_detection_result(broadcast_data)
            
            protocols = ['OSC', 'TCP', 'UDP', 'Socket.IO']
            print(f"📡 [BROADCAST] Sent '{result['match_result']}' to {', '.join(protocols)}")
            
        except Exception as e:
            logger.error(f"Enhanced broadcast error: {e}")
    
    def get_stored_images_info(self) -> Dict[str, Any]:
        """Get information about stored images and capabilities"""
        return {
            'total_stored_images': len(self.stored_images),
            'stored_images': list(self.stored_images.keys()),
            'status': 'ready' if len(self.stored_images) > 0 else 'no_images',
            'threshold': 60.0,
            'matching_methods': ['Traditional (ORB+SIFT)', 'Vector Search (CNN)'],
            'embedding_models': list(self.embedding_extractor.models.keys()),
            'vector_search': {
                'pinecone_available': self.vector_search_engine.pinecone_index is not None,
                'faiss_available': self.vector_search_engine.faiss_index is not None
            }
        }

# Global enhanced matcher instance
enhanced_matcher = None

def get_enhanced_matcher() -> EnhancedObjectMatcher:
    """Get or create enhanced matcher instance"""
    global enhanced_matcher
    if enhanced_matcher is None:
        enhanced_matcher = EnhancedObjectMatcher()
    return enhanced_matcher

def enhanced_object_matching(camera_frame: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Main function: Enhanced object matching with 60% threshold
    Combines traditional computer vision with deep learning embeddings
    """
    matcher = get_enhanced_matcher()
    return matcher.match_with_stored_images(camera_frame)
