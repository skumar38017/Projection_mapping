import os
from dotenv import load_dotenv
from pinecone import Pinecone
import numpy as np
from pathlib import Path
import logging

load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings:
    def __init__(self):
        # Project paths
        self.PROJECT_NAME = os.getenv("PROJECT_NAME", "Random Object Verification")
        self.VERSION = os.getenv("VERSION", "1.0.0")
        
        # Directories
        self.BASE_DIR = Path(__file__).parent
        self.ASSETS_DIR = self.BASE_DIR / "assets"
        self.DEBUG_DIR = self.BASE_DIR / "debug"
        self.STATIC_DIR = self.BASE_DIR / "static"
        
        # Create directories if they don't exist
        self.ASSETS_DIR.mkdir(exist_ok=True)
        self.DEBUG_DIR.mkdir(exist_ok=True)
        self.STATIC_DIR.mkdir(exist_ok=True)
        
        # Pinecone configuration
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "object-verification")
        self.DIMENSION = int(os.getenv("DIMENSION", "1280"))  # EfficientNetB0 output dimension
        
        # Detection thresholds
        self.FEATURE_MATCH_THRESHOLD = float(os.getenv("FEATURE_MATCH_THRESHOLD", "0.3"))
        self.DEEP_MATCH_THRESHOLD = float(os.getenv("DEEP_MATCH_THRESHOLD", "0.7"))
        self.DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.5"))
        
        # Camera settings
        self.DEFAULT_CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
        self.DEFAULT_CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
        self.DEFAULT_FPS = int(os.getenv("FPS", "30"))
        
        # Initialize Pinecone if API key is provided
        self.pc = None
        self.index = None
        if self.PINECONE_API_KEY:
            try:
                self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
                self._setup_pinecone_index()
            except Exception as e:
                logger.warning(f"Failed to initialize Pinecone: {e}")
    
    def _setup_pinecone_index(self):
        """Setup Pinecone index for object embeddings"""
        try:
            # Check if index exists
            existing_indexes = [i['name'] for i in self.pc.list_indexes()]
            
            if self.PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.PINECONE_INDEX_NAME}")
                from pinecone import ServerlessSpec
                self.pc.create_index(
                    name=self.PINECONE_INDEX_NAME,
                    dimension=self.DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.PINECONE_ENVIRONMENT
                    )
                )
            
            # Connect to index
            self.index = self.pc.Index(self.PINECONE_INDEX_NAME)
            logger.info(f"Connected to Pinecone index: {self.PINECONE_INDEX_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to setup Pinecone index: {e}")

# Global settings instance
settings = Settings()
