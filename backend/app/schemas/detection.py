# ~/app/schemas/detection.py

from pydantic import BaseModel
from typing import List, Tuple

class Shape3D(BaseModel):
    shape_type: str
    width: float  # in meters
    height: float  # in meters
    depth: float  # in meters
    vertices_3d: List[Tuple[int, int]] = []  # 3D points in image coordinates

class DetectionResult(BaseModel):
    label: str
    confidence: float
    shape_3d: Shape3D