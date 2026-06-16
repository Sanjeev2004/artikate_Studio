from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def split_into_legal_chunks(
    pages: List[Dict[str, Any]], 
    chunk_size: int = 500, 
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """Splits ingested pages into chunks preserving legal structure (paragraphs).
    
    Args:
        pages: List of pages from load_pdf.
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Target overlap between chunks in characters.
        
    Returns:
        List of chunks: [{"document": str, "page": int, "chunk": str}]
    """
    chunks = []
    
    for page in pages:
        doc_name = page["document"]
        page_num = page["page"]
        text = page["text"]
        
        if not text:
            continue
            
        # Split by paragraphs / clauses to maintain structure
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_len = len(para)
            
            # If paragraph itself is too large, split it by sentences or characters
            if para_len > chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append({
                        "document": doc_name,
                        "page": page_num,
                        "chunk": chunk_text
                    })
                    current_chunk = []
                    current_length = 0
                
                # Split large paragraph
                words = para.split(" ")
                sub_chunk = []
                sub_len = 0
                for word in words:
                    if sub_len + len(word) + 1 > chunk_size:
                        chunks.append({
                            "document": doc_name,
                            "page": page_num,
                            "chunk": " ".join(sub_chunk)
                        })
                        # Retain overlap words
                        overlap_words = sub_chunk[-max(1, int(chunk_overlap / 6)):]
                        sub_chunk = list(overlap_words)
                        sub_len = sum(len(w) + 1 for w in sub_chunk)
                    sub_chunk.append(word)
                    sub_len += len(word) + 1
                if sub_chunk:
                    chunks.append({
                        "document": doc_name,
                        "page": page_num,
                        "chunk": " ".join(sub_chunk)
                    })
            else:
                # If adding this paragraph exceeds chunk_size, emit current chunk
                if current_length + para_len + 2 > chunk_size and current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append({
                        "document": doc_name,
                        "page": page_num,
                        "chunk": chunk_text
                    })
                    
                    # Retain overlap: find how many paragraphs to keep
                    overlap_chunk = []
                    overlap_len = 0
                    for prev_para in reversed(current_chunk):
                        if overlap_len + len(prev_para) + 2 <= chunk_overlap:
                            overlap_chunk.insert(0, prev_para)
                            overlap_len += len(prev_para) + 2
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = overlap_len
                
                current_chunk.append(para)
                current_length += para_len + 2
                
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "document": doc_name,
                "page": page_num,
                "chunk": chunk_text
            })
            
    logger.info(f"Generated {len(chunks)} legal chunks.")
    return chunks
