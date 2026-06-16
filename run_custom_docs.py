import os
import sys

# Ensure the src directory is available in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.section2.pipeline import RAGPipeline

def main():
    docs_dir = "data/my_docs"
    
    # 1. Initialize Pipeline
    print("Initializing RAG Pipeline (Loading embedding model)...")
    pipeline = RAGPipeline()
    
    # 2. Ingest documents
    print(f"\nScanning directory '{docs_dir}' for PDFs...")
    if not os.path.exists(docs_dir) or not [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]:
        print(f"Error: No PDF files found in '{docs_dir}'. Please add some PDFs and run again.")
        return
        
    pipeline.ingest_directory(docs_dir)
    print("Ingestion complete!")
    
    # 3. Interactive query loop
    print("\n" + "="*50)
    print("RAG System Ready! Ask questions about your documents.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)
    
    while True:
        try:
            question = input("\nYour Question: ").strip()
            if question.lower() in ['exit', 'quit']:
                break
            if not question:
                continue
                
            print("\nSearching and generating answer...")
            result = pipeline.query(question)
            
            print("\n--- Answer ---")
            print(result["answer"])
            
            print("\n--- Sources Used ---")
            if result["sources"]:
                for source in result["sources"]:
                    print(f"- {source['document']} (Page {source['page']})")
            else:
                print("- No sources retrieved.")
                
            print(f"\nConfidence Score: {result['confidence']}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
