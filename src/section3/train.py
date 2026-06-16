import os
import sys
import json
import logging

# Ensure project root is in path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle

from src.section2.embeddings import EmbeddingEngine
from src.section3.config import (
    MODEL_DIR,
    MODEL_PATH,
    DATA_DIR,
    NUM_TICKETS_PER_CLASS,
    RANDOM_SEED,
    CV_FOLDS,
    PARAM_GRID
)
from src.section3.dataset import generate_synthetic_tickets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_classifier():
    """Trains the ticket classifier using cross-validation hyperparameter search."""
    logger.info("Generating synthetic training dataset...")
    train_data = generate_synthetic_tickets(NUM_TICKETS_PER_CLASS)
    
    # Save training dataset to disk
    os.makedirs(DATA_DIR, exist_ok=True)
    train_dataset_path = os.path.join(DATA_DIR, "train_dataset.json")
    with open(train_dataset_path, "w") as f:
        json.dump(train_data, f, indent=2)
    logger.info(f"Training dataset saved to {train_dataset_path}")
        
    logger.info("Embedding training texts using Sentence Transformers...")
    embedder = EmbeddingEngine()
    
    texts = [item["text"] for item in train_data]
    labels = [item["label"] for item in train_data]
    
    # Generate embeddings
    X = embedder.embed_documents(texts)
    y = np.array(labels)
    
    logger.info(f"Running Logistic Regression Grid Search (CV={CV_FOLDS}) to find optimal hyper-parameters...")
    base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    
    # Grid search cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(X, y)
    
    logger.info(f"Best parameter found: {grid_search.best_params_}")
    logger.info(f"Best cross-validation accuracy: {grid_search.best_score_:.2%}")
    
    # Fit the best model on the complete set
    best_model = grid_search.best_estimator_
    
    # Save model weights
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"Model saved successfully to {MODEL_PATH}")

if __name__ == "__main__":
    train_classifier()
