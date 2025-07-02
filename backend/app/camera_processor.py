import cv2
import numpy as np
import asyncio
import time  # Added missing import
from typing import Optional, Dict, List
from app.detection.object_detector import ObjectDetector, Detection  # Added Detection import
from app.detection.advanced_shape_analyzer import UniversalShapeReconstructor

class CameraProcessor:
    def __init__(self):
        self.detector = ObjectDetector()
        self.reconstructor = UniversalShapeReconstructor()
        self.last_results = []
        
    async def process_frame(self, frame: np.ndarray, depth: Optional[np.ndarray] = None) -> Dict:
        """Process frame and return 3D reconstruction results"""
        if depth is None:
            depth = self._estimate_depth(frame)
            
        # Detect objects
        detections = await asyncio.get_event_loop().run_in_executor(
            None, self.detector.detect, frame)
        
        # Process each detection
        results = []
        tasks = []
        for det in detections:
            tasks.append(asyncio.get_event_loop().run_in_executor(
                None, self.reconstructor.analyze, frame, depth, det))
        
        # Wait for all reconstructions
        reconstructions = await asyncio.gather(*tasks)
        
        # Prepare output
        output = {
            'timestamp': time.time(),
            'objects': [],
            'visualization': frame.copy()
        }
        
        for det, recon in zip(detections, reconstructions):
            if recon is not None:
                obj_data = {
                    'label': det.label,
                    'confidence': det.confidence,
                    'bbox': det.bbox,
                    'dimensions': recon['dimensions'],
                    'type': recon['type'],
                    'orientation': recon['orientation'].tolist()
                }
                output['objects'].append(obj_data)
                
                # Draw on visualization
                self._draw_result(output['visualization'], det, recon)
        
        self.last_results = output['objects']
        return output

    def _estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Fallback depth estimation when no depth camera available"""
        # Simple depth-from-focus approximation
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.var(laplacian)
        depth = np.ones_like(gray, dtype=np.float32) * (1 - np.clip(variance/1000, 0, 1))
        return depth

    def _draw_result(self, frame: np.ndarray, detection: Detection, reconstruction: dict):
            """Enhanced drawing with polygon outlines and 3D visualization"""
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Draw polygon outline if we have vertices
            if 'vertices_3d' in reconstruction and reconstruction['vertices_3d']:
                vertices = np.array(reconstruction['vertices_3d'], dtype=np.int32)
                
                # Offset vertices by bbox coordinates
                vertices[:, 0] += x1
                vertices[:, 1] += y1
                
                # Draw the convex hull
                cv2.polylines(frame, [vertices], True, (0, 255, 255), 2)
                
                # Draw vertices as circles
                for (x, y) in vertices:
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            
            # Draw label and dimensions
            label = f"{detection.label}: {reconstruction['type']}"
            dims = f"L:{reconstruction['dimensions']['length']:.2f} W:{reconstruction['dimensions']['width']:.2f} H:{reconstruction['dimensions']['height']:.2f}"
            
            cv2.putText(frame, label, (x1, y1-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, dims, (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw orientation axes if available
            if 'orientation' in reconstruction:
                center = ((x1+x2)//2, (y1+y2)//2)
                axes = reconstruction['orientation'] * 50
                for i, color in enumerate([(255,0,0), (0,255,0), (0,0,255)]):
                    end = (int(center[0] + axes[0,i]), int(center[1] + axes[1,i]))
                    cv2.line(frame, center, end, color, 2)