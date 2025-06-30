# ~/app/schemas/detection.py

from pydantic import BaseModel
from typing import List, Tuple, Optional

class Shape3D(BaseModel):
    shape_type: str
    width: float  # in meters
    height: float  # in meters
    depth: float  # in meters
    vertices_3d: List[Tuple[float, float, float]] = []
    mesh_vertices: Optional[List[Tuple[float, float, float]]] = None
    mesh_triangles: Optional[List[Tuple[int, int, int]]] = None

class DetectionResult(BaseModel):
    label: str
    confidence: float
    shape_3d: Shape3D
    mask_points: Optional[List[Tuple[int, int]]] = None