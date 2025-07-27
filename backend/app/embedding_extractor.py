"""
Image Embedding Extractor - CNN Feature Extraction
Supports CLIP, ResNet, MobileNet for high-quality vector embeddings
Simplified version to avoid dependency conflicts
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import logging
import time
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import os
from pathlib import Path

# Deep learning imports
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)

class ImageEmbeddingExtractor:
    """
    Multi-model image embedding extractor with CNN models
    Supports CLIP, ResNet, MobileNet for feature extraction
    """
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.models = {}
        self.transforms = {}
        self.embedding_cache = {}
        
        # Initialize available models
        self._init_clip_model()
        self._init_resnet_model()
        self._init_mobilenet_model()
        self._init_efficientnet_model()
        
        # FAISS index for fast similarity search
        self.faiss_index = None
        self.faiss_id_to_filename = {}
        
        # Pinecone integration
        self.pinecone_index = settings.index if hasattr(settings, 'index') else None
        
        logger.info(f"🧠 ImageEmbeddingExtractor initialized with {len(self.models)} models on {self.device}")
    
    def _get_optimal_device(self):
        """Get optimal device for inference"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("🖥️ Using CPU for embeddings")
        return device
    
    def _init_clip_model(self):
        """Initialize CLIP model for high-quality embeddings"""
        if not CLIP_AVAILABLE:
            logger.warning("CLIP not available")
            return
        
        try:
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            self.models['clip'] = model
            self.transforms['clip'] = preprocess
            
            logger.info("✅ CLIP ViT-B/32 model loaded")
            print("🧠 [CLIP] Vision Transformer loaded for high-quality embeddings")
            
        except Exception as e:
            logger.error(f"CLIP initialization failed: {e}")
    
    def _init_resnet_model(self):
        """Initialize ResNet model for robust feature extraction"""
        try:
            # Load pre-trained ResNet50
            model = models.resnet50(pretrained=True)
            # Remove final classification layer to get features
            model = nn.Sequential(*list(model.children())[:-1])
            model.eval()
            model.to(self.device)
            
            self.models['resnet50'] = model
            
            # Standard ImageNet preprocessing
            self.transforms['resnet50'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("✅ ResNet50 model loaded")
            print("🧠 [ResNet50] Convolutional model loaded for robust features")
            
        except Exception as e:
            logger.error(f"ResNet initialization failed: {e}")
    
    def _init_mobilenet_model(self):
        """Initialize MobileNet for fast, lightweight embeddings"""
        try:
            # Load pre-trained MobileNetV2
            model = models.mobilenet_v2(pretrained=True)
            # Remove classifier to get features
            model.classifier = nn.Identity()
            model.eval()
            model.to(self.device)
            
            self.models['mobilenet_v2'] = model
            
            # MobileNet preprocessing
            self.transforms['mobilenet_v2'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("✅ MobileNetV2 model loaded")
            print("🧠 [MobileNet] Lightweight model loaded for fast inference")
            
        except Exception as e:
            logger.error(f"MobileNet initialization failed: {e}")
    
    def _init_efficientnet_model(self):
        """Initialize EfficientNet for balanced performance"""
        try:
            # Load pre-trained EfficientNet-B0
            model = models.efficientnet_b0(pretrained=True)
            # Remove classifier
            model.classifier = nn.Identity()
            model.eval()
            model.to(self.device)
            
            self.models['efficientnet_b0'] = model
            
            # EfficientNet preprocessing
            self.transforms['efficientnet_b0'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("✅ EfficientNet-B0 model loaded")
            print("🧠 [EfficientNet] Balanced model loaded for optimal performance")
            
        except Exception as e:
            logger.error(f"EfficientNet initialization failed: {e}")
    
    def extract_embedding(self, image: np.ndarray, model_name: str = 'clip') -> np.ndarray:
        """
        Extract embedding from image using specified model
        
        Args:
            image: Input image (BGR format from OpenCV)
            model_name: Model to use ('clip', 'resnet50', 'mobilenet_v2', 'efficientnet_b0')
        
        Returns:
            Normalized embedding vector
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not available, falling back to available models")
            model_name = list(self.models.keys())[0] if self.models else None
            
        if not model_name:
            logger.error("No embedding models available")
            return np.zeros(512)  # Return zero vector
        
        try:
            start_time = time.time()
            
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            if model_name == 'clip':
                embedding = self._extract_clip_embedding(image_rgb)
            else:
                embedding = self._extract_cnn_embedding(image_rgb, model_name)
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            processing_time = time.time() - start_time
            
            print(f"🧠 [EMBEDDING] {model_name.upper()}: {len(embedding)}D vector in {processing_time*1000:.1f}ms")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return np.zeros(512)
    
    def _extract_clip_embedding(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract CLIP embedding"""
        model = self.models['clip']
        preprocess = self.transforms['clip']
        
        # Preprocess image
        image_tensor = preprocess(image_rgb).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            embedding = image_features.cpu().numpy().flatten()
        
        return embedding
    
    def _extract_cnn_embedding(self, image_rgb: np.ndarray, model_name: str) -> np.ndarray:
        """Extract CNN embedding (ResNet, MobileNet, EfficientNet)"""
        model = self.models[model_name]
        transform = self.transforms[model_name]
        
        # Preprocess image
        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = model(image_tensor)
            embedding = features.cpu().numpy().flatten()
        
        return embedding
    
    def extract_multiple_embeddings(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract embeddings using all available models"""
        embeddings = {}
        
        for model_name in self.models.keys():
            embeddings[model_name] = self.extract_embedding(image, model_name)
        
        return embeddings
    
    def get_best_embedding_model(self) -> str:
        """Get the best available embedding model"""
        # Priority order: CLIP > EfficientNet > ResNet > MobileNet
        priority = ['clip', 'efficientnet_b0', 'resnet50', 'mobilenet_v2']
        
        for model_name in priority:
            if model_name in self.models:
                return model_name
        
        return list(self.models.keys())[0] if self.models else None
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about available embedding models"""
        info = {
            'available_models': list(self.models.keys()),
            'device': str(self.device),
            'best_model': self.get_best_embedding_model(),
            'model_details': {}
        }
        
        for model_name in self.models.keys():
            if model_name == 'clip':
                info['model_details'][model_name] = {
                    'type': 'Vision Transformer',
                    'embedding_size': 512,
                    'description': 'High-quality multimodal embeddings'
                }
            elif model_name == 'resnet50':
                info['model_details'][model_name] = {
                    'type': 'Convolutional Neural Network',
                    'embedding_size': 2048,
                    'description': 'Robust feature extraction'
                }
            elif model_name == 'mobilenet_v2':
                info['model_details'][model_name] = {
                    'type': 'Lightweight CNN',
                    'embedding_size': 1280,
                    'description': 'Fast inference for mobile/edge'
                }
            elif model_name == 'efficientnet_b0':
                info['model_details'][model_name] = {
                    'type': 'Efficient CNN',
                    'embedding_size': 1280,
                    'description': 'Balanced performance and accuracy'
                }
        
        return info

# Global embedding extractor instance
embedding_extractor = None

def get_embedding_extractor() -> ImageEmbeddingExtractor:
    """Get or create embedding extractor instance"""
    global embedding_extractor
    if embedding_extractor is None:
        embedding_extractor = ImageEmbeddingExtractor()
    return embedding_extractor
