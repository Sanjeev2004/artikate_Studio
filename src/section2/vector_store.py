import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, dimension: int):
        """Initializes the FAISS IndexFlatIP (cosine similarity if normalized) or IndexFlatL2.
        
        Args:
            dimension: Dimensionality of the embedding vectors.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index
        self.metadata: List[Dict[str, Any]] = []   # Mapping index ID -> metadata chunk
        
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Adds document chunks and their pre-computed embeddings to FAISS index.
        
        Args:
            chunks: List of chunk dicts.
            embeddings: Numpy array of embeddings (normalized for inner product/cosine).
        """
        assert len(chunks) == len(embeddings), "Chunks and embeddings size mismatch"
        if len(embeddings) == 0:
            return
            
        # Normalize embeddings to unit length for Cosine Similarity via Inner Product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        normalized_embeddings = embeddings / norms
        
        self.index.add(normalized_embeddings.astype('float32'))
        self.metadata.extend(chunks)
        logger.info(f"Added {len(chunks)} vectors to index. Total vectors: {self.index.ntotal}")
        
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Searches the index for the top-k nearest chunks.
        
        Args:
            query_embedding: Query embedding vector.
            k: Number of results to retrieve.
            
        Returns:
            List of tuples: (chunk_metadata, cosine_similarity_score)
        """
        if self.index.ntotal == 0:
            return []
            
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        norm = 1.0 if norm == 0 else norm
        normalized_query = (query_embedding / norm).reshape(1, -1).astype('float32')
        
        scores, indices = self.index.search(normalized_query, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            results.append((self.metadata[idx], float(score)))
            
        return results
        
    def save(self, path_prefix: str):
        """Saves index and metadata to disk."""
        faiss.write_index(self.index, f"{path_prefix}.index")
        with open(f"{path_prefix}.meta", "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Vector store saved to prefix {path_prefix}")
            
    def load(self, path_prefix: str):
        """Loads index and metadata from disk."""
        self.index = faiss.read_index(f"{path_prefix}.index")
        with open(f"{path_prefix}.meta", "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"Vector store loaded. Total vectors: {self.index.ntotal}")
