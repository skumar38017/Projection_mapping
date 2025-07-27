# ~/app/utils.py

import base64
import logging
import cv2
import numpy as np
import json
from io import BytesIO
from PIL import Image
from typing import Any

# Configure logger
logger = logging.getLogger("object-verifier")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def b64_to_cv2_image(b64_data: str) -> np.ndarray:
    decoded = base64.b64decode(b64_data)
    np_arr = np.frombuffer(decoded, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def resize_image_cv2(img: np.ndarray, size=(224, 224)) -> np.ndarray:
    return cv2.resize(img, size)

def image_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def draw_bounding_box(img: np.ndarray, label: str = "Matched", color=(0, 255, 0)) -> np.ndarray:
    h, w, _ = img.shape
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), color, 2)
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img

def make_json_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable types to JSON-serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    else:
        return obj

def safe_json_dumps(obj: Any) -> str:
    """Safely serialize object to JSON string, handling numpy types"""
    try:
        serializable_obj = make_json_serializable(obj)
        return json.dumps(serializable_obj)
    except Exception as e:
        logger.error(f"Error serializing to JSON: {e}")
        return json.dumps({"error": "Serialization failed"})
