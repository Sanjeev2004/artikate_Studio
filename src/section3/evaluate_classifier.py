import os
import sys
import json
import logging

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.section3.config import CLASSES, EVAL_REPORT_PATH
from src.section3.classifier import TicketClassifier
from src.section3.dataset import load_eval_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_evaluation():
    """Evaluates the ticket classifier and prints performance metrics."""
    logger.info("Loading evaluation dataset...")
    try:
        eval_set = load_eval_set()
    except FileNotFoundError as e:
        logger.error(f"Error loading evaluation dataset: {e}")
        sys.exit(1)
        
    logger.info("Initializing classifier...")
    classifier = TicketClassifier()
    
    texts = [item["text"] for item in eval_set]
    y_true = [item["label"] for item in eval_set]
    
    logger.info("Running inference on evaluation split...")
    y_pred = classifier.predict_batch(texts)
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    report_str = classification_report(y_true, y_pred, target_names=CLASSES)
    report_dict = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    
    # Display results to console
    print("\n=== Ticket Classifier Evaluation ===")
    print(f"Overall Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(report_str)
    
    print("\nConfusion Matrix:")
    print("Labels order:", CLASSES)
    print(cm)
    
    # Identify most-confused classes
    max_confusion = 0
    confused_pair = ("", "")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j:
                if cm[i][j] > max_confusion:
                    max_confusion = cm[i][j]
                    confused_pair = (CLASSES[i], CLASSES[j])
                    
    print("\nAnalysis of Most-Confused Classes:")
    print(f"The model confuses '{confused_pair[0]}' most often with '{confused_pair[1]}' ({max_confusion} times).")
    
    # Save structured evaluation report to disk
    report_data = {
        "overall_accuracy": acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASSES,
        "most_confused_pair": {
            "class_1": confused_pair[0],
            "class_2": confused_pair[1],
            "count": int(max_confusion)
        }
    }
    
    os.makedirs(os.path.dirname(EVAL_REPORT_PATH), exist_ok=True)
    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Structured evaluation report saved to {EVAL_REPORT_PATH}")

if __name__ == "__main__":
    run_evaluation()
