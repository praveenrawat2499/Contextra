import os
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    store = FaissVectorStore("faiss_store")
    
    # Build the store if files don't exist, otherwise load it
    faiss_path = os.path.join("faiss_store", "faiss.index")
    meta_path = os.path.join("faiss_store", "metadata.pkl")
    if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
        docs = load_all_documents("data")
        store.build_from_documents(docs)
    else:
        store.load()
        
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "What is Linear Regression ?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
