# Artikate Studio — AI / ML / LLM Engineer Assessment

My submission for the four required sections. Written reasoning lives in
[ANSWERS.md](ANSWERS.md), the RAG architecture notes in [DESIGN.md](DESIGN.md).

- Section 1 (diagnosing a failing pipeline): written, in ANSWERS.md
- Section 2 (RAG pipeline): code in `src/section2/`, design in DESIGN.md
- Section 3 (ticket classifier): code in `src/section3/`, justification in ANSWERS.md
- Section 4 (systems design review): written, in ANSWERS.md (questions A, B, C)

## Layout

```
.
├── README.md
├── DESIGN.md              # Section 2 architecture + trade-offs
├── ANSWERS.md             # Sections 1, 3, 4 written answers
├── requirements.txt
├── src/
│   ├── section2/          # RAG: ingestion, chunking, embeddings, FAISS, pipeline, eval
│   └── section3/          # classifier, training, evaluation
└── tests/
    └── test_latency.py    # asserts <500ms/ticket inference
```

## Setup

Needs Python 3.11+. First run downloads the `all-MiniLM-L6-v2` embedding model from
Hugging Face (~90MB, one time), so the very first command needs internet; after that
everything runs offline.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
.venv\Scripts\pip install -r requirements.txt
```

### API keys (optional)

The RAG pipeline calls an LLM for the final answer generation if a key is set, and
otherwise falls back to a local extractive generator, so the whole thing runs end-to-end
with no key. To use a real LLM, set one of:

```powershell
$env:OPENAI_API_KEY = "..."   # uses gpt-4o-mini
$env:GEMINI_API_KEY = "..."   # uses gemini-2.5-flash
```

Retrieval, the Precision@3 score, refusal behaviour, and the Section 3 classifier are all
fully local and don't depend on a key.

## Running it

Each script works run directly or as a module, whichever you prefer.

### Section 2 — RAG

Generates 3 sample legal PDFs, builds the FAISS index, runs the 10 hand-written eval
questions, prints Precision@3, and shows the refusal path on an unanswerable question.

```powershell
.venv\Scripts\python src\section2\evaluate.py
# or: .venv\Scripts\python -m src.section2.evaluate
```

Expected: `Precision@3 Score: 0.90 (9/10)` and a refused answer for the
favorite-color question.

### Section 3 — ticket classifier

```powershell
# 1. train (generates synthetic data, fits the LR head over embeddings)
.venv\Scripts\python src\section3\train.py

# 2. evaluate (accuracy, per-class F1, confusion matrix on 100 hand-written tickets)
.venv\Scripts\python src\section3\evaluate_classifier.py

# 3. latency test (20 tickets, asserts each prediction is valid and <500ms)
.venv\Scripts\pytest tests\test_latency.py -s
```

Expected: ~92% accuracy, and the latency test passing at ~30ms/ticket.
