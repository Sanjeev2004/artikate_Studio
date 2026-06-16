import os
import json
import random
import pickle
import numpy as np
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.section2.embeddings import EmbeddingEngine
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Allowed classes
CLASSES = ["billing", "technical_issue", "feature_request", "complaint", "other"]

def generate_synthetic_tickets(num_per_class: int = 200) -> List[Dict[str, str]]:
    """Generates synthetic support tickets based on structural templates."""
    random.seed(42)
    
    templates = {
        "billing": [
            "I was charged twice for the same subscription in {month}.",
            "Can I get a refund for my payment on {date}?",
            "My credit card was declined. How do I update my billing info?",
            "Why did the subscription price increase this month?",
            "I need a copy of my invoice for {month}.",
            "Please cancel my premium billing renewal.",
            "I see a charge of {amount} from your company that I don't recognize.",
            "How do I switch from monthly to annual billing?",
            "Is there a discount code I can apply to my invoice?",
            "The checkout page says payment failed but money was deducted."
        ],
        "technical_issue": [
            "The export to CSV button does nothing when I click it.",
            "I am getting a 500 error when trying to log in.",
            "The app crashes every time I upload a file larger than {size}.",
            "Why is the page loading so slowly today?",
            "I cannot connect my account to the server.",
            "The dashboard is showing a blank white screen.",
            "My password reset link is broken or expired.",
            "The search bar is not returning any results for valid queries.",
            "I'm stuck in an infinite redirect loop on the homepage.",
            "The mobile app does not sync data with the desktop version."
        ],
        "feature_request": [
            "It would be great if you could add a dark mode option.",
            "Do you plan to integrate with Slack or Microsoft Teams?",
            "Please implement multi-factor authentication for better security.",
            "Can we get a bulk export feature for user logs?",
            "I want the ability to customize dashboard widgets.",
            "Is there a way to schedule automated reports weekly?",
            "Please add support for Markdown formatting in the editor.",
            "Would love to see a Gantt chart view in the project board.",
            "Could you support uploading HEIC image formats?",
            "Please create an API endpoint to retrieve usage statistics."
        ],
        "complaint": [
            "Your customer support is extremely slow and unhelpful.",
            "This is the worst experience I have ever had with software.",
            "The new update has ruined the user interface. I hate it.",
            "I am very disappointed with the lack of communication.",
            "This bug has been open for three weeks. This is unacceptable.",
            "Your service has been down twice this week during business hours.",
            "I feel cheated by the false advertising on your website.",
            "The agent I spoke with was incredibly rude to me.",
            "I am going to leave a 1-star review on Product Hunt.",
            "I want to speak with a manager immediately."
        ],
        "other": [
            "Hello, I just wanted to say thank you for the great product!",
            "Are you guys hiring for remote software engineering roles?",
            "Where can I find the documentation for your Javascript SDK?",
            "Can I use your logo on my company's partners page?",
            "What are your support team's holiday office hours?",
            "I am a student working on a thesis, can I interview someone?",
            "Do you have a public roadmap for product updates?",
            "I would like to submit a partnership proposal.",
            "Can you tell me more about your company's carbon footprint policy?",
            "Just checking if my test email went through successfully."
        ]
    }
    
    # Fillers to create diversity
    months = ["January", "February", "March", "April", "May", "June", "December"]
    dates = ["yesterday", "last Friday", "May 15th", "two days ago"]
    amounts = ["$15", "$49", "$99", "$299"]
    sizes = ["5MB", "10MB", "50MB", "100MB"]
    
    dataset = []
    for label, patterns in templates.items():
        for i in range(num_per_class):
            pattern = random.choice(patterns)
            # format with fillers if placeholder exists
            text = pattern.format(
                month=random.choice(months),
                date=random.choice(dates),
                amount=random.choice(amounts),
                size=random.choice(sizes)
            )
            # Add some minor noise/phrases to increase diversity
            prefixes = ["", "Hi, ", "Help: ", "Quick question: ", "URGENT: "]
            text = random.choice(prefixes) + text
            dataset.append({"text": text, "label": label})
            
    return dataset

def train_classifier():
    """Trains a classifier on top of Sentence Transformer embeddings."""
    logger.info("Generating synthetic training dataset...")
    train_data = generate_synthetic_tickets(200) # 1000 examples
    
    # Save training dataset
    os.makedirs("data/tickets", exist_ok=True)
    with open("data/tickets/train_dataset.json", "w") as f:
        json.dump(train_data, f, indent=2)
        
    logger.info("Embedding training texts using Sentence Transformers...")
    embedder = EmbeddingEngine()
    
    texts = [item["text"] for item in train_data]
    labels = [item["label"] for item in train_data]
    
    X = embedder.embed_documents(texts)
    y = np.array(labels)
    
    logger.info("Training Logistic Regression classifier...")
    # L2 regularized Logistic Regression is highly stable and fast
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X, y)
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/ticket_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved successfully to {model_path}")
    
if __name__ == "__main__":
    train_classifier()
