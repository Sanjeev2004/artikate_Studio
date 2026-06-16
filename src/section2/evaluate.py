import os
import sys
import logging

# Let this script run either way: `python src/section2/evaluate.py` or `python -m src.section2.evaluate`.
# When run directly, only the script's folder is on sys.path, so the `src` package isn't importable
# until we add the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from src.section2.pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_sample_pdfs(output_dir: str):
    """Generates 3 sample legal PDF documents using ReportLab."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    body_style = styles["Normal"]
    
    # 1. NDA with Vendor X
    nda_path = os.path.join(output_dir, "NDA_Vendor_X.pdf")
    doc1 = SimpleDocTemplate(nda_path, pagesize=letter)
    story1 = [
        Paragraph("MUTUAL NON-DISCLOSURE AGREEMENT", title_style),
        Spacer(1, 12),
        Paragraph("This Non-Disclosure Agreement ('Agreement') is entered into on January 10, 2026, by and between Company A and Vendor X.", body_style),
        Spacer(1, 12),
        Paragraph("1. Purpose: The parties wish to explore a business relationship and share proprietary intellectual property.", body_style),
        Spacer(1, 12),
        Paragraph("2. Confidential Information: Includes all technical, financial, and product pricing details disclosed by either party.", body_style),
        Spacer(1, 12),
        Paragraph("3. Notice Period for Termination: Either party may terminate this agreement upon giving a Notice Period of thirty (30) days written notice to the other party.", body_style),
        Spacer(1, 12),
        Paragraph("4. Limitation of Liability: Neither party's total liability under this NDA shall exceed the sum of INR 50 lakhs (INR 5,000,000).", body_style),
    ]
    doc1.build(story1)
    logger.info(f"Generated {nda_path}")
    
    # 2. SOW Consulting Agreement
    sow_path = os.path.join(output_dir, "SOW_Consulting.pdf")
    doc2 = SimpleDocTemplate(sow_path, pagesize=letter)
    story2 = [
        Paragraph("STATEMENT OF WORK: CONSULTING SERVICES", title_style),
        Spacer(1, 12),
        Paragraph("This Statement of Work is effective as of February 1, 2026, by and between Company A and Consultant Y.", body_style),
        Spacer(1, 12),
        Paragraph("1. Scope of Work: Consultant Y will provide AI system integration and pipeline optimization services.", body_style),
        Spacer(1, 12),
        Paragraph("2. Notice Period: This engagement can be terminated by either party upon a Notice Period of fifteen (15) days written notice.", body_style),
        Spacer(1, 12),
        Paragraph("3. Limitation of Liability: The total liability of Consultant Y for any claim arising out of this contract shall be capped at INR 1.5 crore (INR 15,000,000).", body_style),
    ]
    doc2.build(story2)
    logger.info(f"Generated {sow_path}")
    
    # 3. Employment Agreement
    emp_path = os.path.join(output_dir, "Employment_Agreement.pdf")
    doc3 = SimpleDocTemplate(emp_path, pagesize=letter)
    story3 = [
        Paragraph("EMPLOYMENT AGREEMENT", title_style),
        Spacer(1, 12),
        Paragraph("This Employment Agreement is dated March 1, 2026, between Company A and Employee Z.", body_style),
        Spacer(1, 12),
        Paragraph("1. Position and Duties: Employee Z is hired as a Senior Machine Learning Engineer.", body_style),
        Spacer(1, 12),
        Paragraph("2. Notice Period: The employee must provide a Notice Period of ninety (90) days prior to resignation.", body_style),
        Spacer(1, 12),
        Paragraph("3. Limitation of Liability: The employee's liability for willful misconduct shall be capped at INR 10 lakhs (INR 1,000,000).", body_style),
    ]
    doc3.build(story3)
    logger.info(f"Generated {emp_path}")

def run_evaluation():
    """Runs the Precision@3 evaluation for the RAG pipeline."""
    data_dir = "data/legal_docs"
    generate_sample_pdfs(data_dir)
    
    # Initialize and ingest pipeline
    pipeline = RAGPipeline(similarity_threshold=0.25)
    pipeline.ingest_directory(data_dir)
    
    # Define 10 manually written QA pairs with expected source documents
    eval_set = [
        {
            "question": "What is the notice period in the NDA with Vendor X?",
            "expected_doc": "NDA_Vendor_X.pdf"
        },
        {
            "question": "Which contract has a notice period of 15 days?",
            "expected_doc": "SOW_Consulting.pdf"
        },
        {
            "question": "What is the limitation of liability in the consulting agreement?",
            "expected_doc": "SOW_Consulting.pdf"
        },
        {
            "question": "Is there any agreement signed on January 10, 2026?",
            "expected_doc": "NDA_Vendor_X.pdf"
        },
        {
            "question": "What is the notice period for Employee Z?",
            "expected_doc": "Employment_Agreement.pdf"
        },
        {
            "question": "What is the liability cap in the NDA with Vendor X?",
            "expected_doc": "NDA_Vendor_X.pdf"
        },
        {
            "question": "Who is the contracting consultant in SOW_Consulting.pdf?",
            "expected_doc": "SOW_Consulting.pdf"
        },
        {
            "question": "What is the notice period in the employment agreement?",
            "expected_doc": "Employment_Agreement.pdf"
        },
        {
            "question": "Which contract has a limitation of liability above INR 1 crore?",
            "expected_doc": "SOW_Consulting.pdf"
        },
        {
            "question": "Does Vendor X's NDA mention intellectual property?",
            "expected_doc": "NDA_Vendor_X.pdf"
        }
    ]
    
    hits = 0
    print("\n--- RAG Evaluation (Precision@3) ---")
    for idx, item in enumerate(eval_set):
        q = item["question"]
        expected = item["expected_doc"]
        
        result = pipeline.query(q)
        sources = result["sources"]
        
        # Check if expected document is in the top 3 retrieved sources
        retrieved_docs = [s["document"] for s in sources]
        is_hit = expected in retrieved_docs
        
        if is_hit:
            hits += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        print(f"Q{idx+1}: '{q}'")
        print(f"  Expected: {expected} | Retrieved: {retrieved_docs} | Status: {status}")
        print(f"  Confidence: {result['confidence']} | Answer: {result['answer'][:100]}...\n")
        
    precision_at_3 = hits / len(eval_set)
    print(f"Precision@3 Score: {precision_at_3:.2f} ({hits}/{len(eval_set)})")
    
    # Test hallucination mitigation/refusal
    print("\n--- Testing Hallucination Refusal ---")
    unrelated_q = "What is the favorite color of Vendor X's CEO?"
    refusal_result = pipeline.query(unrelated_q)
    print(f"Q: '{unrelated_q}'")
    print(f"Answer: {refusal_result['answer']}")
    print(f"Sources retrieved: {len(refusal_result['sources'])} | Confidence: {refusal_result['confidence']}")
    
    return precision_at_3

if __name__ == "__main__":
    run_evaluation()
