import time
import pytest
from src.section3.classifier import TicketClassifier

# List of 20 raw ticket strings to test
TEST_TICKETS = [
    "I need a refund for the billing issue on my credit card statement.",
    "The page is completely broken on my browser, it just crashes.",
    "Could we get a export to pdf feature added to the team interface?",
    "Your support is horrible. I've been waiting for hours.",
    "Hi, is there any discount for academic usage?",
    "Why was I billed $100 instead of $10?",
    "The API is throwing 504 gateway timeout errors today.",
    "Please add support for Google Calendar sync.",
    "This application is a waste of time and money, very buggy.",
    "How do I apply for the open senior position?",
    "Where can I access my billing receipt for May?",
    "The login form freezes whenever I try to log in on mobile.",
    "Can you build a feature that lets users comment on reports?",
    "I am extremely angry about the system downtime today.",
    "Is your service compliant with GDPR rules?",
    "How do I upgrade to the annual plan?",
    "The search query doesn't match the results at all.",
    "It would be awesome if you supported dark mode.",
    "This is the worst customer service experience of my life.",
    "Just wanted to drop a line and say your product is nice!"
]

def test_inference_latency_and_validity():
    classifier = TicketClassifier()
    allowed_classes = {"billing", "technical_issue", "feature_request", "complaint", "other"}
    
    total_time = 0.0
    predictions = []
    
    print("\n--- Running Latency Test (20 Tickets) ---")
    for text in TEST_TICKETS:
        start_time = time.perf_counter()
        pred = classifier.predict(text)
        end_time = time.perf_counter()
        
        latency = (end_time - start_time) * 1000  # In milliseconds
        total_time += latency
        predictions.append(pred)
        
        print(f"Ticket: '{text[:40]}...' -> Pred: {pred} | Latency: {latency:.2f}ms")
        
        # Verify prediction is valid
        assert pred in allowed_classes, f"Predicted class {pred} is not one of the allowed classes."
        # Verify single prediction latency is well within 500ms
        assert latency < 500.0, f"Single ticket prediction exceeded 500ms: {latency:.2f}ms"
        
    avg_latency = total_time / len(TEST_TICKETS)
    print(f"\nAverage Latency per ticket: {avg_latency:.2f}ms")
    print(f"Total time for 20 tickets: {total_time:.2f}ms")
    
    # Assert average latency is way under 500ms
    assert avg_latency < 500.0, f"Average latency exceeded 500ms: {avg_latency:.2f}ms"
