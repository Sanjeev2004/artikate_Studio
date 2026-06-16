import os
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initializes the SentenceTransformer embedding model.
        
        Args:
            model_name: Hugging Face model name.
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts into vector representations.
        
        Args:
            texts: List of document/chunk texts.
            
        Returns:
            A numpy array of embeddings (shape: num_texts, embedding_dim).
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single query string.
        
        Args:
            text: Query text.
            
        Returns:
            A numpy array of the embedding vector.
        """
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
