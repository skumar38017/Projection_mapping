# ~/app/detection/advanced_shape_analyzer.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image
import open3d as o3d
from scipy.spatial import Delaunay
from typing import List, Optional, Tuple
from .object_detector import Detection
from ..schemas.detection import Shape3D

class AdvancedShapeAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_segmentation_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Camera intrinsics (should be calibrated)
        self.fx = 1000  # Focal length x
        self.fy = 1000  # Focal length y
        self.cx = 640   # Principal point x
        self.cy = 360   # Principal point y

    def _load_segmentation_model(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        return model.to(self.device).eval()

    def analyze(self, frame: np.ndarray, depth: np.ndarray, detection: Detection) -> Optional[Shape3D]:
        """Analyze object and return precise 3D shape"""
        x1, y1, x2, y2 = detection.bbox
        roi = frame[y1:y2, x1:x2]
        
        # 1. Get precise mask
        mask = self._get_object_mask(roi)
        if mask is None:
            return None
            
        # 2. Create 3D point cloud
        points_3d = self._create_point_cloud(roi, depth[y1:y2, x1:x2], mask)
        
        # 3. Create mesh
        mesh = self._create_mesh(points_3d)
        
        # 4. Calculate dimensions
        dimensions = self._calculate_dimensions(points_3d)
        
        return Shape3D(
            shape_type="object",
            width=dimensions['width'],
            height=dimensions['height'],
            depth=dimensions['depth'],
            vertices_3d=points_3d.tolist(),
            mesh_vertices=mesh.vertices.tolist() if mesh else [],
            mesh_triangles=mesh.triangles.tolist() if mesh else []
        )

    def _get_object_mask(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Get precise object mask using segmentation"""
        input_tensor = self.transform(roi).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        output = F.softmax(output, dim=0).cpu().numpy()
        
        # Get the most likely class (excluding background)
        mask = np.argmax(output, axis=0)
        if np.max(mask) == 0:  # Only background
            return None
            
        # Refine mask
        mask = (mask > 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), iterations=2))
        
        # Get precise boundary
        contour = find_contours(mask, 0.5)[0]
        hull = convex_hull_image(mask)
        combined = np.maximum(mask, hull.astype(np.uint8))
        
        return combined

    def _create_point_cloud(self, roi: np.ndarray, depth_roi: np.ndarray, 
                          mask: np.ndarray) -> np.ndarray:
        """Create 3D point cloud from depth and mask"""
        points = []
        height, width = roi.shape[:2]
        
        for y in range(height):
            for x in range(width):
                if mask[y, x] > 0 and depth_roi[y, x] > 0:
                    # Convert to 3D coordinates
                    z = depth_roi[y, x]
                    x_3d = (x - self.cx) * z / self.fx
                    y_3d = (y - self.cy) * z / self.fy
                    points.append([x_3d, y_3d, z])
        
        return np.array(points)

    def _create_mesh(self, points_3d: np.ndarray) -> Optional[o3d.geometry.TriangleMesh]:
        """Create mesh from 3D points using Poisson reconstruction"""
        if len(points_3d) < 4:
            return None
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # Estimate normals
        pcd.estimate_normals()
        
        # Create mesh
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        return mesh

    def _calculate_dimensions(self, points_3d: np.ndarray) -> dict:
        """Calculate object dimensions from point cloud"""
        if len(points_3d) == 0:
            return {'width': 0, 'height': 0, 'depth': 0}
            
        min_coords = np.min(points_3d, axis=0)
        max_coords = np.max(points_3d, axis=0)
        
        return {
            'width': abs(max_coords[0] - min_coords[0]),
            'height': abs(max_coords[1] - min_coords[1]),
            'depth': abs(max_coords[2] - min_coords[2])
        }