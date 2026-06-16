# Artikate Studio — AI / ML / LLM Engineer Assessment

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FAISS](https://img.shields.io/badge/VectorStore-FAISS-green)
![SentenceTransformers](https://img.shields.io/badge/Embeddings-Sentence_Transformers-orange)
![Status](https://img.shields.io/badge/Status-Completed_&_Verified-brightgreen)

Welcome to my submission for the Artikate Studio technical assessment. This repository contains a complete, production-grade implementation for all four required sections. 

The architecture is built with a focus on **low latency, offline capability, and hallucination prevention**, meeting all constraints (such as the <500ms CPU latency limit).

---

## Optional Section 5: Live System Walkthrough

Watch the complete live system demonstration here:  
**[Watch the System Walkthrough Video](https://www.tella.tv/video/rag-app-ticket-classifier-demo-bau2)**

---

## Assessment Completion Checklist

| Section | Description | Status | Reference File |
|---|---|---|---|
| **Section 1** | Diagnose a Failing LLM Pipeline | Completed | [`ANSWERS.md`](ANSWERS.md) |
| **Section 2** | Build a Production-Grade RAG Pipeline | Completed | `src/section2/`, [`DESIGN.md`](DESIGN.md) |
| **Section 3** | Ticket Classifier (<500ms on CPU) | Completed | `src/section3/`, `tests/` |
| **Section 4** | Written Systems Design Review | Completed | [`ANSWERS.md`](ANSWERS.md) |

---

## Repository Structure

```text
.
├── README.md               # Setup and execution instructions
├── DESIGN.md               # Architecture and trade-off reasoning for RAG pipeline
├── ANSWERS.md              # Written answers for Sections 1, 3, and 4
├── requirements.txt        # Dependency definitions
├── src/
│   ├── section2/           # RAG: ingestion, chunking, embeddings, FAISS, pipeline, eval
│   └── section3/           # Classifier: training, inference, evaluation
├── tests/
│   └── test_latency.py     # Pytest assertion for the <500ms latency constraint
└── data/                   # Automatically generated legal PDFs and ticket datasets
```

---

## Setup & Installation

This project is built to run flawlessly in a clean Python 3.11+ environment. 

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90MB). After this one-time download, **all critical operations (Retrieval, Classification, Metrics) run 100% offline.**

### 1. Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
.venv\Scripts\pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
The RAG pipeline is designed to be **fail-safe**. If no API keys are provided, it automatically falls back to a local, rule-based extractive generator so you can still test it end-to-end.

To use a real LLM for Generation, set one of the following environment variables:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-key-here"  # Uses gpt-4o-mini
# OR
$env:GEMINI_API_KEY="your-key-here"  # Uses gemini-2.5-flash
```

---

## Running the Code

### Section 2: Production RAG Pipeline
This script automatically generates 3 sample legal PDFs, chunks them, builds the FAISS index, and runs an evaluation harness on 10 hand-crafted questions.

```powershell
$env:PYTHONPATH="." 
.venv\Scripts\python src\section2\evaluate.py
```
> **Expected Output:** You will see the pipeline score a **Precision@3 of 0.90 (9/10)**. It will also explicitly demonstrate the refusal behavior on a query lacking context.

### Section 3: Ticket Classifier
This section runs locally using transformer embeddings + a logistic regression head to ensure ultra-low latency.

**1. Train the Classifier** (Generates 1000 synthetic tickets and fits the model)
```powershell
$env:PYTHONPATH="." 
.venv\Scripts\python src\section3\train.py
```

**2. Evaluate Accuracy & Confusion Matrix** (On 100 held-out examples)
```powershell
$env:PYTHONPATH="." 
.venv\Scripts\python src\section3\evaluate_classifier.py
```
> **Expected Output:** ~92% accuracy with a detailed report on the most confused classes.

**3. Run the Latency Constraint Test** (Verifies the <500ms requirement)
```powershell
$env:PYTHONPATH="." 
.venv\Scripts\pytest tests\test_latency.py -s -v
```
> **Expected Output:** The test passes with an average inference time of **~30ms per ticket** (well under the 500ms limit).

---
*If you have any questions during the review, please refer to the detailed explanations in [DESIGN.md](DESIGN.md) and [ANSWERS.md](ANSWERS.md).*
