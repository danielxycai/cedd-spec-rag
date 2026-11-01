import os
import shutil
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.ingest import load_and_chunk_specification

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    """Initializes a lightweight local embedding model running on CPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def build_vector_store(chunks, persist_dir: str = CHROMA_DIR, force_reindex: bool = False):
    """Builds or loads a persistent local ChromaDB instance."""
    embeddings = get_embeddings()

    if force_reindex and os.path.exists(persist_dir):
        print(f"Removing old vectorstore at {persist_dir}...")
        shutil.rmtree(persist_dir)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"Loading existing ChromaDB from {persist_dir}...")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        print(f"Creating new ChromaDB vectorstore at {persist_dir}...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        print("Vectorstore indexing complete.")

    return vectorstore

def build_hybrid_retriever(chunks, vectorstore, k: int = 4):
    """
    Builds an EnsembleRetriever combining:
    1. ChromaDB vector similarity (dense semantic search)
    2. BM25 keyword matching (sparse lexical search)
    """
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid_retriever

if __name__ == "__main__":
    print("--- Testing Indexing & Hybrid Retrieval ---")
    chunks = load_and_chunk_specification()
    
    vectorstore = build_vector_store(chunks, force_reindex=True)
    hybrid_retriever = build_hybrid_retriever(chunks, vectorstore, k=3)

    test_query = "What are the requirements for bending steel reinforcement?"
    print(f"\nQuerying: '{test_query}'")
    results = hybrid_retriever.invoke(test_query)

    print(f"\nRetrieved {len(results)} chunks:")
    for i, doc in enumerate(results, 1):
        clause = doc.metadata.get("clause_id", "Unknown")
        page = doc.metadata.get("page", "?")
        print(f"\n[{i}] Source: {clause} (Page {page})")
        print(f"Snippet: {doc.page_content[:200]}...")