import os
import sys
import pickle
import numpy as np
from typing import List

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.section2.embeddings import EmbeddingEngine
from src.section3.config import MODEL_PATH

class TicketClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        """Loads the embedding engine and the pre-trained classification head."""
        self.embedding_engine = EmbeddingEngine()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. Please run training first via: "
                "python src/section3/train.py"
            )
            
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
            
    def predict(self, text: str) -> str:
        """Predicts the category of a single support ticket.
        
        Args:
            text: Ticket raw string.
            
        Returns:
            The predicted class label (one of billing, technical_issue, feature_request, complaint, other).
        """
        embedding = self.embedding_engine.embed_query(text)
        embedding = embedding.reshape(1, -1)
        pred = self.model.predict(embedding)[0]
        return str(pred)
        
    def predict_batch(self, texts: List[str]) -> List[str]:
        """Predicts categories for a batch of support tickets.
        
        Args:
            texts: List of ticket raw strings.
            
        Returns:
            List of predicted class labels.
        """
        embeddings = self.embedding_engine.embed_documents(texts)
        preds = self.model.predict(embeddings)
        return [str(p) for p in preds]
