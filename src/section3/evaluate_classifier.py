import os
import sys
import json
import logging

# Allow running directly (python src/section3/evaluate_classifier.py) as well as via `python -m`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.section3.classifier import TicketClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASSES = ["billing", "technical_issue", "feature_request", "complaint", "other"]

# 100 verified/realistic evaluation examples (20 per class)
EVAL_SET = [
    # BILLING (20 examples)
    {"text": "I was charged $49 yesterday but my account shows free tier.", "label": "billing"},
    {"text": "Please refund the renewal charge from last week.", "label": "billing"},
    {"text": "My subscription failed because my card expired. How do I pay?", "label": "billing"},
    {"text": "Why did my monthly bill increase without any notification?", "label": "billing"},
    {"text": "I need to download invoices from June 2025.", "label": "billing"},
    {"text": "Cancel my automatic subscription renewal immediately.", "label": "billing"},
    {"text": "There is a double charge on my bank statement for March.", "label": "billing"},
    {"text": "How do I upgrade to the corporate plan and pay via invoice?", "label": "billing"},
    {"text": "I used a coupon code but the system charged me full price.", "label": "billing"},
    {"text": "My company requires VAT details on all receipts.", "label": "billing"},
    {"text": "The transaction failed but my credit card was still debited.", "label": "billing"},
    {"text": "I want to change my credit card details for billing.", "label": "billing"},
    {"text": "How do I dispute a charge on my subscription?", "label": "billing"},
    {"text": "What are the pricing tiers for the API service?", "label": "billing"},
    {"text": "Are there any hidden fees for overage bandwidth usage?", "label": "billing"},
    {"text": "Is it possible to pay annually instead of monthly?", "label": "billing"},
    {"text": "Why was I charged taxes when my business is tax-exempt?", "label": "billing"},
    {"text": "Please delete my billing profile and credit card history.", "label": "billing"},
    {"text": "Where is the checkout page? I want to buy premium.", "label": "billing"},
    {"text": "My trial ended and I got billed automatically without warning.", "label": "billing"},

    # TECHNICAL ISSUE (20 examples)
    {"text": "The CSV export button is completely unresponsive.", "label": "technical_issue"},
    {"text": "I am getting a internal server error 500 when saving.", "label": "technical_issue"},
    {"text": "The app keeps crashing when uploading heavy PDF documents.", "label": "technical_issue"},
    {"text": "The page takes over 15 seconds to load on Chrome.", "label": "technical_issue"},
    {"text": "Cannot sync my workspace data, getting network error.", "label": "technical_issue"},
    {"text": "The entire layout looks broken on my Safari browser.", "label": "technical_issue"},
    {"text": "I didn't receive the password reset email after clicking reset.", "label": "technical_issue"},
    {"text": "The search function returns zero results for existant files.", "label": "technical_issue"},
    {"text": "I am locked out because of a login loop error.", "label": "technical_issue"},
    {"text": "The integration is failing to fetch new webhooks.", "label": "technical_issue"},
    {"text": "Every time I click submit, the page simply refreshes.", "label": "technical_issue"},
    {"text": "Images are not rendering on the dashboard anymore.", "label": "technical_issue"},
    {"text": "The API is returning a 401 unauthorized error for my active key.", "label": "technical_issue"},
    {"text": "Why is my database connection dropping every few minutes?", "label": "technical_issue"},
    {"text": "I can't delete a project, it says permission denied but I am admin.", "label": "technical_issue"},
    {"text": "The text editor is lagging heavily when typing.", "label": "technical_issue"},
    {"text": "Session expires too quickly, logging me out every 5 minutes.", "label": "technical_issue"},
    {"text": "I found a security vulnerability in your API parameters.", "label": "technical_issue"},
    {"text": "The system is showing incorrect timezones for all event logs.", "label": "technical_issue"},
    {"text": "My local docker container cannot connect to your cloud DB.", "label": "technical_issue"},

    # FEATURE REQUEST (20 examples)
    {"text": "It would be great if you could add a dark mode.", "label": "feature_request"},
    {"text": "Can we get integration with Slack to receive notifications?", "label": "feature_request"},
    {"text": "Please support two-factor authentication for added safety.", "label": "feature_request"},
    {"text": "I need a way to bulk export my team settings.", "label": "feature_request"},
    {"text": "Can we customize the colors of the dashboard widgets?", "label": "feature_request"},
    {"text": "Please allow scheduling automated reports every Friday.", "label": "feature_request"},
    {"text": "We need Markdown styling support in the comments section.", "label": "feature_request"},
    {"text": "Is there a roadmap for adding Gantt charts to projects?", "label": "feature_request"},
    {"text": "Please add HEIC image format support to uploads.", "label": "feature_request"},
    {"text": "We need an API endpoint to fetch team usage statistics.", "label": "feature_request"},
    {"text": "Can you add a folder structure to organize files?", "label": "feature_request"},
    {"text": "It would be nice to have a mobile app for offline editing.", "label": "feature_request"},
    {"text": "Could you add a print button to format reports nicely?", "label": "feature_request"},
    {"text": "Is it possible to integrate with Google Drive?", "label": "feature_request"},
    {"text": "Please add search functionality inside PDF files.", "label": "feature_request"},
    {"text": "We would like to customize the email templates sent to clients.", "label": "feature_request"},
    {"text": "Can we have role-based permissions for folders?", "label": "feature_request"},
    {"text": "Please support exporting data to Excel format.", "label": "feature_request"},
    {"text": "I request an option to auto-save drafts every few seconds.", "label": "feature_request"},
    {"text": "Could you implement keyboard shortcuts for navigation?", "label": "feature_request"},

    # COMPLAINT (20 examples)
    {"text": "The customer support team is extremely slow and useless.", "label": "complaint"},
    {"text": "This app is the absolute worst, it keeps breaking my flow.", "label": "complaint"},
    {"text": "The new update ruined the dashboard UI. I hate it.", "label": "complaint"},
    {"text": "Very disappointed by the lack of updates and bad support.", "label": "complaint"},
    {"text": "This bug has been open for a month, this is terrible.", "label": "complaint"},
    {"text": "Service was down twice during my client presentation.", "label": "complaint"},
    {"text": "The false advertising on your homepage is very dishonest.", "label": "complaint"},
    {"text": "The service agent was extremely rude and unhelpful to me.", "label": "complaint"},
    {"text": "I am leaving a 1-star review on all rating platforms.", "label": "complaint"},
    {"text": "Your company is refusing to address my data privacy concerns.", "label": "complaint"},
    {"text": "The software is completely unusable after the latest patch.", "label": "complaint"},
    {"text": "I have been waiting for a response to my email for 4 days.", "label": "complaint"},
    {"text": "Your pricing model is a total rip-off for small startups.", "label": "complaint"},
    {"text": "Why do I have to pay extra for basic features? Fraudulent.", "label": "complaint"},
    {"text": "I want to delete my account and speak with a supervisor.", "label": "complaint"},
    {"text": "This tool is a waste of money, I regret buying it.", "label": "complaint"},
    {"text": "Your documentation is full of errors and outdated guidelines.", "label": "complaint"},
    {"text": "No one is helping me with my locked account. Disgraceful.", "label": "complaint"},
    {"text": "The interface is too cluttered and gives me a headache.", "label": "complaint"},
    {"text": "I was promised a refund and now your team is ignoring me.", "label": "complaint"},

    # OTHER (20 examples)
    {"text": "Hi, just wanted to say thanks for the awesome product!", "label": "other"},
    {"text": "Are you hiring remote frontend developers at the moment?", "label": "other"},
    {"text": "Where is the documentation for the Python client library?", "label": "other"},
    {"text": "Can we use your company name on our partner section?", "label": "other"},
    {"text": "What are your customer service hours for holidays?", "label": "other"},
    {"text": "I'm a student doing research, can I interview a PM?", "label": "other"},
    {"text": "Do you have a public roadmap that we can look at?", "label": "other"},
    {"text": "I'd like to pitch a marketing partnership proposal.", "label": "other"},
    {"text": "What is your data retention policy for EU users?", "label": "other"},
    {"text": "Just verifying if my registration email worked.", "label": "other"},
    {"text": "When is your next product webinar scheduled?", "label": "other"},
    {"text": "Can you speak at our tech conference next month?", "label": "other"},
    {"text": "Do you offer any discounts for non-profit organizations?", "label": "other"},
    {"text": "Where is your corporate headquarters located?", "label": "other"},
    {"text": "I am looking for your brand guidelines and logos.", "label": "other"},
    {"text": "Do you support single sign-on for corporate domains?", "label": "other"},
    {"text": "Just sending a friendly hello to the team!", "label": "other"},
    {"text": "Is there an archive of your company newsletter?", "label": "other"},
    {"text": "I want to pitch an article for your engineering blog.", "label": "other"},
    {"text": "What is the contact email for press inquiries?", "label": "other"}
]

def run_evaluation():
    """Evaluates the ticket classifier and prints performance metrics."""
    # Save test dataset
    os.makedirs("data/tickets", exist_ok=True)
    with open("data/tickets/eval_dataset.json", "w") as f:
        json.dump(EVAL_SET, f, indent=2)
        
    logger.info("Initializing classifier...")
    classifier = TicketClassifier()
    
    texts = [item["text"] for item in EVAL_SET]
    y_true = [item["label"] for item in EVAL_SET]
    
    logger.info("Running inference on evaluation split...")
    y_pred = classifier.predict_batch(texts)
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    
    print("\n=== Ticket Classifier Evaluation ===")
    print(f"Overall Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(report)
    
    print("\nConfusion Matrix:")
    print("Labels order:", CLASSES)
    print(cm)
    
    # Identify most-confused classes
    # Look at off-diagonals in the confusion matrix
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
    
if __name__ == "__main__":
    run_evaluation()
