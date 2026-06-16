import os
import json
import random
import logging
from typing import List, Dict, Any
from src.section3.config import (
    TEMPLATES_JSON_PATH,
    EVAL_SET_JSON_PATH,
    NUM_TICKETS_PER_CLASS,
    RANDOM_SEED
)

logger = logging.getLogger(__name__)

def load_templates() -> Dict[str, List[str]]:
    """Loads the synthetic ticket templates from the JSON data file."""
    if not os.path.exists(TEMPLATES_JSON_PATH):
        raise FileNotFoundError(f"Templates file not found at {TEMPLATES_JSON_PATH}.")
    with open(TEMPLATES_JSON_PATH, "r") as f:
        return json.load(f)

def load_eval_set() -> List[Dict[str, str]]:
    """Loads the hand-written evaluation dataset from the JSON data file."""
    if not os.path.exists(EVAL_SET_JSON_PATH):
        raise FileNotFoundError(f"Evaluation dataset file not found at {EVAL_SET_JSON_PATH}.")
    with open(EVAL_SET_JSON_PATH, "r") as f:
        return json.load(f)

def generate_synthetic_tickets(num_per_class: int = NUM_TICKETS_PER_CLASS) -> List[Dict[str, str]]:
    """Generates synthetic support tickets using the templates JSON file and random fillers.
    
    Args:
        num_per_class: Number of synthetic tickets to generate per class category.
        
    Returns:
        A list of dictionaries containing 'text' and 'label' for each ticket.
    """
    templates = load_templates()
    random.seed(RANDOM_SEED)
    
    # Fillers to inject variety into templates
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    dates = ["yesterday", "last Friday", "May 15th", "two days ago", "last week", "this morning"]
    amounts = ["$15", "$29", "$49", "$99", "$199", "$299"]
    sizes = ["5MB", "10MB", "20MB", "50MB", "100MB", "500MB"]
    prefixes = ["", "Hi, ", "Help: ", "Quick question: ", "URGENT: ", "Please help: "]
    
    dataset = []
    for label, patterns in templates.items():
        for _ in range(num_per_class):
            pattern = random.choice(patterns)
            
            # Safely format placeholders if they exist in the pattern
            text = pattern.format(
                month=random.choice(months),
                date=random.choice(dates),
                amount=random.choice(amounts),
                size=random.choice(sizes)
            )
            
            # Inject prefix to simulate realistic user writing variations
            prefix = random.choice(prefixes)
            text = prefix + text
            
            dataset.append({"text": text, "label": label})
            
    return dataset
