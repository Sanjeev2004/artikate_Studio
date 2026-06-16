import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from dotenv import load_dotenv

# Load API keys from .env file if present
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env')))
from src.section2.document_ingestion import load_pdf
from src.section2.chunking import split_into_legal_chunks
from src.section2.embeddings import EmbeddingEngine
from src.section2.vector_store import VectorStore

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2", 
        chunk_size: int = 500, 
        chunk_overlap: int = 100,
        similarity_threshold: float = 0.25
    ):
        """Initializes RAG Pipeline components."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        
        self.embedding_engine = EmbeddingEngine(embedding_model)
        # dimension for all-MiniLM-L6-v2 is 384
        self.vector_store = VectorStore(dimension=384)
        
    def ingest_directory(self, data_dir: str):
        """Loads and indexes all PDF files in the directory."""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        all_pages = []
        for file in os.listdir(data_dir):
            if file.endswith(".pdf"):
                path = os.path.join(data_dir, file)
                all_pages.extend(load_pdf(path))
                
        if not all_pages:
            logger.warning("No PDF files found to ingest.")
            return
            
        chunks = split_into_legal_chunks(all_pages, self.chunk_size, self.chunk_overlap)
        texts = [c["chunk"] for c in chunks]
        embeddings = self.embedding_engine.embed_documents(texts)
        self.vector_store.add_documents(chunks, embeddings)
        logger.info("Ingestion and indexing complete.")
        
    def _generate_answer_local(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Fallback rule-based answer generator when no API key is available."""
        # Simple extraction and synthesis of sentences related to key terms in the query
        keywords = [w.lower() for w in re.findall(r'\w+', question) if len(w) > 3]
        best_sentences = []
        
        for c in retrieved_chunks:
            chunk_text = c["chunk"]
            sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
            for sentence in sentences:
                score = sum(1 for kw in keywords if kw in sentence.lower())
                if score > 0:
                    best_sentences.append((score, sentence.strip()))
                    
        if best_sentences:
            best_sentences.sort(key=lambda x: x[0], reverse=True)
            # Take top 3 unique sentences
            seen = set()
            unique_sentences = []
            for score, sent in best_sentences:
                if sent not in seen:
                    seen.add(sent)
                    unique_sentences.append(sent)
                if len(unique_sentences) >= 3:
                    break
            
            # Format as clean bullet points
            formatted_ans = "Based on the retrieved documents, here are the key references found:\n\n"
            for sentence in unique_sentences:
                formatted_ans += f"- {sentence}\n"
            return formatted_ans
        
        # Default fallback if keywords don't match
        top_chunk = retrieved_chunks[0]['chunk'][:300].replace('\n', ' ').strip()
        return f"Based on the retrieved context:\n\n- {top_chunk}..."

    def _generate_answer_llm(self, question: str, context: str) -> Optional[str]:
        """Calls external APIs (Groq, OpenAI, or Gemini) if keys are provided."""
        
        system_prompt = (
            "You are a precise legal document QA assistant. "
            "Answer the user's question using ONLY the provided context. "
            "Cite the source document and page number in your answer. "
            "If the answer is not in the context, say: 'I cannot find the answer in the provided context.'"
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        # Check Groq (Free, fast, Llama 3.3 70B)
        if os.environ.get("GROQ_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=os.environ["GROQ_API_KEY"],
                    base_url="https://api.groq.com/openai/v1"
                )
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Groq Generation error: {e}")

        # Check OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI Generation error: {e}")
                
        # Check Google Gemini
        if os.environ.get("GEMINI_API_KEY"):
            try:
                from google import genai
                client = genai.Client()
                prompt = f"{system_prompt}\n\n{user_prompt}"
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini Generation error: {e}")
                
        return None


    def query(self, question: str) -> Dict[str, Any]:
        """Queries the RAG pipeline.
        
        Args:
            question: Question string.
            
        Returns:
            Dict: {"answer": str, "sources": List[Dict], "confidence": float}
        """
        query_embedding = self.embedding_engine.embed_query(question)
        retrieved = self.vector_store.search(query_embedding, k=3)
        
        if not retrieved:
            return {
                "answer": "No documents ingested. Please ingest documents first.",
                "sources": [],
                "confidence": 0.0
            }
            
        # Check top similarity score for refusal threshold
        top_chunk, top_score = retrieved[0]
        if top_score < self.similarity_threshold:
            return {
                "answer": "I am sorry, but the retrieved context does not contain enough information to answer this question.",
                "sources": [],
                "confidence": float(round(top_score, 2))
            }
            
        # Grounding check: verify overlap of question keywords with retrieved context
        context_text = "\n\n".join([c[0]["chunk"] for c in retrieved])
        
        # Generation
        llm_answer = self._generate_answer_llm(question, context_text)
        is_fallback = False
        if llm_answer:
            answer = llm_answer
        else:
            # Fallback to local rule-based sentence generator
            answer = self._generate_answer_local(question, [c[0] for c in retrieved])
            is_fallback = True
            
        # Perform grounding check on the generated answer
        # Refuse to answer if LLM explicitly says it cannot find the answer
        refusal_phrases = [
            "i cannot find", "i am sorry", "insufficient context", 
            "not in the provided context", "does not contain information"
        ]
        if any(phrase in answer.lower() for phrase in refusal_phrases):
            return {
                "answer": "I am sorry, but the retrieved context does not contain enough information to answer this question.",
                "sources": [],
                "confidence": float(round(top_score * 0.5, 2)),
                "fallback": False
            }
            
        # Construct sources list
        sources = []
        for chunk_meta, score in retrieved:
            sources.append({
                "document": chunk_meta["document"],
                "page": chunk_meta["page"],
                "chunk": chunk_meta["chunk"]
            })
            
        # Confidence score based on retrieval similarity score
        # Cosine similarity typically ranges from 0.0 to 1.0 here (FlatIP normalized)
        confidence = float(np.clip(top_score, 0.0, 1.0))
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": float(round(confidence, 2)),
            "fallback": is_fallback
        }

    def clear(self):
        """Clears the underlying vector store."""
        self.vector_store.clear()

