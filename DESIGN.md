# DESIGN

Architecture and trade-off notes for the Section 2 RAG pipeline, with the Section 3
classifier choices at the end.

---

## Section 2 — RAG pipeline

```
Legal PDFs
  -> page-aware ingestion (pypdf, page numbers kept)
  -> clause/paragraph chunking
  -> embeddings (all-MiniLM-L6-v2)
  -> FAISS index (inner product on normalized vectors = cosine)
  -> top-k retrieval + similarity-threshold gate
  -> answer generation grounded in retrieved chunks
  -> answer + per-chunk citations (document, page) + confidence
```

### Chunking

Paragraph/clause-aware splitting, target 500 characters with 100 characters of overlap.

The reason I didn't just do fixed-size character windows: legal text is structured into
numbered clauses, and the answer to a question usually lives inside one clause. A blind
character cut tends to slice through the middle of "the notice period is thirty (30)
days", which is exactly the span you need to retrieve intact. So the chunker splits on
paragraph boundaries (`\n\n`) first and only falls back to a word-level overlapping split
when a single paragraph is longer than the target size. The overlap keeps a clause from
being orphaned across a boundary. I kept chunks fairly small because these clauses are
short and self-contained; small chunks give tighter retrieval and cleaner citations than
large multi-clause blocks.

### Embedding model

`sentence-transformers/all-MiniLM-L6-v2`. 384 dimensions, 22M params, runs on CPU in a
few ms per chunk. It's trained for semantic search, so it handles "notice period" vs.
"termination notice" type paraphrase well, and it has no API dependency or per-call cost.
The corpus here is English, which is exactly its strong suit. If retrieval quality on
dense legal language turned out to be the bottleneck I'd move up to a larger e5 / bge
model, but MiniLM is the right starting point for this size.

### Vector store: FAISS

I picked FAISS over the alternatives:
- vs. Pinecone: Pinecone is a hosted cloud service. The brief leans toward something that
  runs locally (and a legal-document use case often can't ship contracts to a third-party
  index), so a managed SaaS is the wrong fit here.
- vs. Chroma: Chroma is pleasant for prototyping but adds a layer I don't need. FAISS is a
  C++ library doing in-memory vector math directly, it's the standard for this, and it has
  a clean path to scale (IVF/HNSW/PQ indexes) when the corpus grows.

For this size I use `IndexFlatIP` — exact search, no approximation, on vectors I normalize
to unit length so inner product equals cosine similarity.

### Retrieval

Top-k cosine, k=3, with a threshold gate at 0.25. Naive top-k is enough at three documents
and ten eval questions; hybrid search and reranking are what I'd add at scale (see below),
not something this corpus needs. The threshold is the important part: if the best chunk
scores below 0.25, nothing relevant was found, so the pipeline refuses instead of
answering from weak context.

### Hallucination mitigation

Three layers, all implemented in `pipeline.py`, not just described:
1. Refusal on weak retrieval. If the top chunk's similarity is below 0.25, return a fixed
   "not enough information" message with confidence dropped, before any generation. This is
   what stops the system answering questions the corpus can't support — e.g. "What's the
   favorite color of Vendor X's CEO?" gets refused.
2. Grounded generation. The LLM prompt restricts the answer to the provided context and is
   told to say it can't find the answer if it isn't there. If the model emits that refusal,
   the pipeline propagates a refusal rather than a low-quality guess.
3. Confidence tied to retrieval. The returned confidence comes from the top chunk's
   similarity score, and collapses toward 0 on a refusal, so a caller can act on a weak
   answer instead of trusting everything equally.

The measured Precision@3 on the 10-question eval set is 0.90 (9/10). The one miss is a
retrieval ranking issue, not a hallucination — the right document is present but not in the
top 3 for that phrasing.

### Scaling from 500 to 50,000 documents (~2M pages)

What breaks, and the fix for each:
1. Index in RAM. `IndexFlatIP` holds every vector in memory and scans all of them per
   query. At ~2M chunks of 384-dim float32 that's on the order of 10–16GB and search goes
   linear. Fix: switch to an approximate index — HNSW for low-latency in-memory, or IVF-PQ
   to compress vectors and cut memory hard while keeping sub-linear search. Accept a small
   recall hit for a large speed/memory win.
2. Ingestion time. Embedding and parsing 50k PDFs serially takes hours. Fix: parallelize
   extraction and embedding (multiprocessing / a batch job), embed on GPU if available, and
   make ingestion incremental so new contracts don't trigger a full rebuild.
3. Retrieval precision. With 50k documents, many clauses look alike and pure vector top-k
   starts returning near-duplicates. Fix: hybrid retrieval (BM25 for exact terms like party
   names and clause numbers, combined with dense vectors), then a cross-encoder reranker
   (e.g. `ms-marco-MiniLM-L-6-v2`) over the top ~20 to pick the final 3. Legal questions
   often hinge on exact identifiers, which is precisely where lexical + rerank beats dense
   alone.

---

## Section 3 — ticket classifier

Approach: frozen `all-MiniLM-L6-v2` sentence embeddings with a logistic-regression head on
top. Not an end-to-end fine-tune — see ANSWERS.md for why a frozen encoder plus a linear
head beats full fine-tuning on 1,000 synthetic examples.

| | Local embedding + LR head | Few-shot LLM API |
|---|---|---|
| Latency (CPU) | ~30ms measured (17–54ms range) | 0.8–2.5s, fails the 500ms limit |
| Cost | $0/day | per-call, accumulates |
| Dependency | none, fully local | network + API key |
| Fit on 1,000 examples | strong (92% held-out) | decent but generic |

The latency constraint alone (500ms on CPU) rules out the API route, and the local model
clears it with ~16x headroom while costing nothing to run. Full numbers, confusion
analysis, and the throughput math are in ANSWERS.md.
