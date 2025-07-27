"""
Vector Search System - Pinecone + FAISS Integration
High-performance similarity search for image embeddings
Supports both cloud (Pinecone) and local (FAISS) vector databases
"""

import numpy as np
import json
import time
import logging
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import pickle

# Vector database imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from app.config import settings
from app.embedding_extractor import get_embedding_extractor

logger = logging.getLogger(__name__)

class VectorSearchEngine:
    """
    Hybrid vector search engine using Pinecone (cloud) and FAISS (local)
    Provides fast similarity search for image embeddings
    """
    
    def __init__(self):
        self.embedding_extractor = get_embedding_extractor()
        
        # Pinecone setup
        self.pinecone_client = None
        self.pinecone_index = None
        self._init_pinecone()
        
        # FAISS setup
        self.faiss_index = None
        self.faiss_id_to_metadata = {}
        self.faiss_index_path = settings.BASE_DIR / "faiss_index"
        self.faiss_index_path.mkdir(exist_ok=True)
        self._init_faiss()
        
        # Reference embeddings storage
        self.reference_embeddings = {}
        self.embedding_metadata = {}
        
        # Load existing embeddings
        self._load_reference_embeddings()
        
        logger.info(f"🔍 VectorSearchEngine initialized")
        print(f"🔍 [VECTOR_SEARCH] Pinecone: {'✅' if self.pinecone_index else '❌'}")
        print(f"🔍 [VECTOR_SEARCH] FAISS: {'✅' if FAISS_AVAILABLE else '❌'}")
    
    def _init_pinecone(self):
        """Initialize Pinecone vector database"""
        if not PINECONE_AVAILABLE:
            logger.warning("Pinecone not available")
            return
        
        try:
            if settings.PINECONE_API_KEY:
                self.pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
                self.pinecone_index = self.pinecone_client.Index(settings.PINECONE_INDEX_NAME)
                
                # Test connection
                stats = self.pinecone_index.describe_index_stats()
                logger.info(f"✅ Pinecone connected: {stats['total_vector_count']} vectors")
                print(f"☁️ [PINECONE] Connected to '{settings.PINECONE_INDEX_NAME}' with {stats['total_vector_count']} vectors")
            else:
                logger.warning("Pinecone API key not provided")
                
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            self.pinecone_index = None
    
    def _init_faiss(self):
        """Initialize FAISS local vector index"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available")
            return
        
        try:
            # Try to load existing FAISS index
            index_file = self.faiss_index_path / "image_embeddings.index"
            metadata_file = self.faiss_index_path / "metadata.pkl"
            
            if index_file.exists() and metadata_file.exists():
                self.faiss_index = faiss.read_index(str(index_file))
                with open(metadata_file, 'rb') as f:
                    self.faiss_id_to_metadata = pickle.load(f)
                
                logger.info(f"✅ FAISS index loaded: {self.faiss_index.ntotal} vectors")
                print(f"💾 [FAISS] Loaded local index with {self.faiss_index.ntotal} vectors")
            else:
                # Create new FAISS index
                # Use 512 dimensions for CLIP, will be adjusted based on actual embeddings
                dimension = 512
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                logger.info("✅ FAISS index created")
                print("💾 [FAISS] Created new local index")
                
        except Exception as e:
            logger.error(f"FAISS initialization failed: {e}")
            self.faiss_index = None
    
    def _load_reference_embeddings(self):
        """Load and generate embeddings for reference images"""
        assets_dir = settings.ASSETS_DIR
        if not assets_dir.exists():
            logger.warning(f"Assets directory not found: {assets_dir}")
            return
        
        print(f"\n🧠 [EMBEDDINGS] Loading reference images from {assets_dir}")
        
        for filename in os.listdir(assets_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self._process_reference_image(filename)
        
        # Save FAISS index
        self._save_faiss_index()
        
        logger.info(f"✅ Loaded embeddings for {len(self.reference_embeddings)} reference images")
    
    def _process_reference_image(self, filename: str):
        """Process a single reference image and generate embeddings"""
        filepath = settings.ASSETS_DIR / filename
        
        try:
            import cv2
            image = cv2.imread(str(filepath))
            if image is None:
                logger.warning(f"Could not load image: {filename}")
                return
            
            print(f"   📸 [PROCESSING] {filename}")
            
            # Extract embeddings using multiple models
            embeddings = self.embedding_extractor.extract_multiple_embeddings(image)
            
            # Store embeddings
            self.reference_embeddings[filename] = embeddings
            
            # Metadata
            metadata = {
                'filename': filename,
                'filepath': str(filepath),
                'timestamp': datetime.now().isoformat(),
                'image_shape': image.shape,
                'embedding_models': list(embeddings.keys())
            }
            self.embedding_metadata[filename] = metadata
            
            # Add to Pinecone
            self._add_to_pinecone(filename, embeddings, metadata)
            
            # Add to FAISS
            self._add_to_faiss(filename, embeddings, metadata)
            
            print(f"   ✅ [EMBEDDED] {filename} with {len(embeddings)} models")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
    
    def _add_to_pinecone(self, filename: str, embeddings: Dict[str, np.ndarray], metadata: Dict):
        """Add embeddings to Pinecone index"""
        if not self.pinecone_index:
            return
        
        try:
            # Use the best embedding model for Pinecone
            best_model = self.embedding_extractor.get_best_embedding_model()
            if best_model not in embeddings:
                return
            
            embedding = embeddings[best_model]
            
            # Prepare vector for Pinecone
            vector_data = {
                'id': f"{filename}_{best_model}",
                'values': embedding.tolist(),
                'metadata': {
                    **metadata,
                    'model': best_model,
                    'embedding_size': len(embedding)
                }
            }
            
            # Upsert to Pinecone
            self.pinecone_index.upsert([vector_data])
            
        except Exception as e:
            logger.error(f"Error adding to Pinecone: {e}")
    
    def _add_to_faiss(self, filename: str, embeddings: Dict[str, np.ndarray], metadata: Dict):
        """Add embeddings to FAISS index"""
        if not self.faiss_index:
            return
        
        try:
            # Use the best embedding model for FAISS
            best_model = self.embedding_extractor.get_best_embedding_model()
            if best_model not in embeddings:
                return
            
            embedding = embeddings[best_model]
            
            # Ensure embedding is the right dimension
            if self.faiss_index.d != len(embedding):
                # Recreate index with correct dimension
                self.faiss_index = faiss.IndexFlatIP(len(embedding))
                self.faiss_id_to_metadata = {}
            
            # Add to FAISS
            embedding_2d = embedding.reshape(1, -1).astype(np.float32)
            self.faiss_index.add(embedding_2d)
            
            # Store metadata
            vector_id = self.faiss_index.ntotal - 1
            self.faiss_id_to_metadata[vector_id] = {
                **metadata,
                'model': best_model,
                'embedding_size': len(embedding)
            }
            
        except Exception as e:
            logger.error(f"Error adding to FAISS: {e}")
    
    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        if not self.faiss_index:
            return
        
        try:
            index_file = self.faiss_index_path / "image_embeddings.index"
            metadata_file = self.faiss_index_path / "metadata.pkl"
            
            faiss.write_index(self.faiss_index, str(index_file))
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.faiss_id_to_metadata, f)
            
            logger.info(f"✅ FAISS index saved with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def search_similar_images(self, query_image: np.ndarray, top_k: int = 5, 
                            use_pinecone: bool = True, use_faiss: bool = True) -> Dict[str, Any]:
        """
        Search for similar images using vector similarity
        
        Args:
            query_image: Input image to search for
            top_k: Number of top results to return
            use_pinecone: Whether to use Pinecone search
            use_faiss: Whether to use FAISS search
        
        Returns:
            Search results with similarity scores
        """
        start_time = time.time()
        
        print(f"\n🔍 [VECTOR_SEARCH] Searching for similar images...")
        
        # Extract embedding from query image
        best_model = self.embedding_extractor.get_best_embedding_model()
        query_embedding = self.embedding_extractor.extract_embedding(query_image, best_model)
        
        results = {
            'query_processed_at': datetime.now().isoformat(),
            'embedding_model': best_model,
            'embedding_size': len(query_embedding),
            'pinecone_results': [],
            'faiss_results': [],
            'combined_results': []
        }
        
        # Search in Pinecone
        if use_pinecone and self.pinecone_index:
            pinecone_results = self._search_pinecone(query_embedding, top_k)
            results['pinecone_results'] = pinecone_results
            print(f"☁️ [PINECONE] Found {len(pinecone_results)} results")
        
        # Search in FAISS
        if use_faiss and self.faiss_index:
            faiss_results = self._search_faiss(query_embedding, top_k)
            results['faiss_results'] = faiss_results
            print(f"💾 [FAISS] Found {len(faiss_results)} results")
        
        # Combine and rank results
        combined_results = self._combine_search_results(
            results['pinecone_results'], 
            results['faiss_results']
        )
        results['combined_results'] = combined_results[:top_k]
        
        processing_time = time.time() - start_time
        results['processing_time_ms'] = processing_time * 1000
        
        print(f"🎯 [SEARCH_COMPLETE] Found {len(combined_results)} matches in {processing_time*1000:.1f}ms")
        
        return results
    
    def _search_pinecone(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Search in Pinecone index"""
        try:
            response = self.pinecone_index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            results = []
            for match in response['matches']:
                results.append({
                    'id': match['id'],
                    'score': float(match['score']),
                    'filename': match['metadata'].get('filename', ''),
                    'model': match['metadata'].get('model', ''),
                    'source': 'pinecone'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Pinecone search error: {e}")
            return []
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Search in FAISS index"""
        try:
            if self.faiss_index.ntotal == 0:
                return []
            
            # Normalize query embedding for cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            query_2d = query_norm.reshape(1, -1).astype(np.float32)
            
            # Search
            scores, indices = self.faiss_index.search(query_2d, min(top_k, self.faiss_index.ntotal))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx in self.faiss_id_to_metadata:
                    metadata = self.faiss_id_to_metadata[idx]
                    results.append({
                        'id': f"faiss_{idx}",
                        'score': float(score),
                        'filename': metadata.get('filename', ''),
                        'model': metadata.get('model', ''),
                        'source': 'faiss'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []
    
    def _combine_search_results(self, pinecone_results: List[Dict], 
                              faiss_results: List[Dict]) -> List[Dict]:
        """Combine and rank results from multiple sources"""
        combined = {}
        
        # Add Pinecone results
        for result in pinecone_results:
            filename = result['filename']
            if filename not in combined or result['score'] > combined[filename]['score']:
                combined[filename] = result
        
        # Add FAISS results
        for result in faiss_results:
            filename = result['filename']
            if filename not in combined or result['score'] > combined[filename]['score']:
                combined[filename] = result
        
        # Sort by score (descending)
        sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        
        return sorted_results
    
    def get_best_match(self, query_image: np.ndarray, threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """
        Get the best matching image with 60% threshold
        
        Args:
            query_image: Input image to match
            threshold: Similarity threshold (0.6 = 60%)
        
        Returns:
            (filename, similarity_score) or (None, 0.0) if no match
        """
        results = self.search_similar_images(query_image, top_k=1)
        
        if results['combined_results']:
            best_match = results['combined_results'][0]
            similarity_score = best_match['score']
            
            # Convert similarity score to percentage
            similarity_percentage = similarity_score * 100
            
            print(f"🎯 [BEST_MATCH] {best_match['filename']}: {similarity_percentage:.1f}% similarity")
            
            if similarity_score >= threshold:
                return best_match['filename'], similarity_score
        
        return None, 0.0
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector search system"""
        stats = {
            'pinecone': {
                'available': self.pinecone_index is not None,
                'index_name': settings.PINECONE_INDEX_NAME if self.pinecone_index else None,
                'vector_count': 0
            },
            'faiss': {
                'available': self.faiss_index is not None,
                'vector_count': self.faiss_index.ntotal if self.faiss_index else 0,
                'dimension': self.faiss_index.d if self.faiss_index else 0
            },
            'reference_images': len(self.reference_embeddings),
            'embedding_models': list(self.embedding_extractor.models.keys()),
            'best_model': self.embedding_extractor.get_best_embedding_model()
        }
        
        # Get Pinecone stats
        if self.pinecone_index:
            try:
                pinecone_stats = self.pinecone_index.describe_index_stats()
                stats['pinecone']['vector_count'] = pinecone_stats['total_vector_count']
            except:
                pass
        
        return stats

# Global vector search engine instance
vector_search_engine = None

def get_vector_search_engine() -> VectorSearchEngine:
    """Get or create vector search engine instance"""
    global vector_search_engine
    if vector_search_engine is None:
        vector_search_engine = VectorSearchEngine()
    return vector_search_engine
