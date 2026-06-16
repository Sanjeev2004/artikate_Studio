import os

# Paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ticket_classifier.pkl")
DATA_DIR = os.path.join("data", "tickets")
TEMPLATES_JSON_PATH = os.path.join(DATA_DIR, "templates.json")
EVAL_SET_JSON_PATH = os.path.join(DATA_DIR, "eval_set.json")
EVAL_REPORT_PATH = os.path.join(DATA_DIR, "evaluation_report.json")

# Classes / Categories
CLASSES = ["billing", "technical_issue", "feature_request", "complaint", "other"]

# Machine Learning Parameters
NUM_TICKETS_PER_CLASS = 200
RANDOM_SEED = 42
CV_FOLDS = 5
PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0]
}
