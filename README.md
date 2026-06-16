# Artikate Studio AI/ML/LLM Engineer Assessment

This repository contains a complete, production-quality implementation of the Artikate Studio AI/ML/LLM Engineer Assessment.

---

## Repository Structure

```
/
├── README.md              # Setup instructions and run commands
├── DESIGN.md              # Architectural decisions and trade-off analysis
├── ANSWERS.md             # Written diagnosis logs and systems design answers
├── requirements.txt       # Project dependencies
├── src/
│   ├── section2/          # Production RAG pipeline files
│   │   ├── document_ingestion.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── pipeline.py
│   │   └── evaluate.py
│   └── section3/          # Ticket classification files
│       ├── classifier.py
│       ├── train.py
│       └── evaluate_classifier.py
├── tests/
│   └── test_latency.py    # Latency assertion test (<500ms on CPU)
└── data/                  # Ingested data (contracts and tickets)
```

---

## Setup Instructions

Ensure you have **Python 3.11+** installed.

### 1. Initialize Virtual Environment & Install Dependencies
Run the following commands in your shell:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install requirements
.venv\Scripts\pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
The RAG pipeline automatically falls back to a **local keyword-matching and extraction engine** if no API keys are provided, allowing full local execution. To run using live LLMs, set either:
```bash
# For OpenAI
$env:OPENAI_API_KEY="your-openai-api-key"

# For Google Gemini
$env:GEMINI_API_KEY="your-gemini-api-key"
```

---

## Execution Commands

### Section 2: Production RAG Pipeline
This script automatically generates 3 legal PDFs, builds the FAISS vector index, runs 10 evaluation QA pairs, computes the **Precision@3** score, and tests the context-refusal mitigation strategy.
```bash
.venv\Scripts\python src/section2/evaluate.py
```

### Section 3: Ticket Classification
1. **Train Model**: Generates synthetic training datasets and trains the classifier head over Sentence Transformer embeddings in under 1 second.
```bash
.venv\Scripts\python src/section3/train.py
```
2. **Evaluate Model**: Computes Accuracy, F1-scores, and outputs the Confusion Matrix on a held-out test split of 100 manually verified tickets.
```bash
.venv\Scripts\python src/section3/evaluate_classifier.py
```
3. **Run Latency Test**: Verifies that prediction latency for 20 tickets is strictly under the 500ms constraint.
```bash
.venv\Scripts\pytest tests/test_latency.py
```
