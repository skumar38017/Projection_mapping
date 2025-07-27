# ~/app/embedding_generator.py

# app/embedding_generator.py
import os
import numpy as np
import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path

assets_path = Path(__file__).parent / "assets"

# Load EfficientNet model for embeddings
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def generate_embedding(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    embedding = model.predict(img)
    return embedding.flatten()
