import os
import faiss
from app.feature_extractor import extract_features
import numpy as np
from pathlib import Path
import logging

# Configure logger
logger = logging.getLogger(__name__)

feature_db = None
image_paths = []

def load_reference_cache():
    global feature_db, image_paths
    
    try:
        # Get the absolute path to the assets directory
        current_dir = Path(__file__).parent
        assets_dir = current_dir / "assets"
        
        # Create directory if it doesn't exist
        assets_dir.mkdir(exist_ok=True)
        
        features = []
        for file in sorted(os.listdir(assets_dir)):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = assets_dir / file
                try:
                    vec = extract_features(str(path))
                    features.append(vec)
                    # Store relative path for web access
                    image_paths.append(file)
                    logger.info(f"Loaded image: {file}")
                except Exception as e:
                    logger.error(f"Error processing {file}: {e}")
                    continue
        
        if not features:
            logger.warning("No valid images found in assets directory")
            return
        
        feature_matrix = np.vstack(features).astype("float32")
        feature_db = faiss.IndexFlatL2(feature_matrix.shape[1])
        feature_db.add(feature_matrix)
        logger.info(f"Loaded {len(features)} reference images into cache")
        
    except Exception as e:
        logger.error(f"Failed to load reference cache: {e}")
        raise