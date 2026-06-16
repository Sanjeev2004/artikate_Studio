import os
from typing import List, Dict, Any
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Loads a PDF and returns a list of pages with text and metadata.
    
    Args:
        file_path: Absolute path to the PDF file.
        
    Returns:
        A list of dicts: [{"document": str, "page": int, "text": str}]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")
        
    filename = os.path.basename(file_path)
    logger.info(f"Ingesting document: {filename}")
    
    pages = []
    try:
        reader = PdfReader(file_path)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({
                "document": filename,
                "page": page_idx + 1,  # 1-based page index
                "text": text.strip()
            })
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise e
        
    return pages
