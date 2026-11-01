import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure workspace root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.ingest import load_and_chunk_specification
from src.indexer import build_vector_store, build_hybrid_retriever

load_dotenv()

SYSTEM_PROMPT = """You are an expert Civil Engineering Assistant specializing in Hong Kong's CEDD General Specification for Civil Engineering Works (Section 15: Steel Reinforcement).

Answer the question strictly based on the provided CEDD context below. Follow these rules:
1. If the context contains the answer, provide a professional, clear, and structured response (use bullet points where appropriate).
2. Cite the exact CEDD Clause number (e.g., Clause 15.12(2), Clause 15.19(1), Clause 15.30(1)) for every requirement stated.
3. If the context does NOT contain the answer at all, simply state: "The provided CEDD Section 15 context does not contain sufficient information to answer this question." 
4. DO NOT append any disclaimers, fallback statements, or notes about "other requirements" if you have successfully provided an answer.
5. WARNING: Do not confuse Section 15 page numbers (e.g., 15.5, 15.7) with clause numbers. Valid Section 15 clauses are strictly numbered between 15.01 and 15.40.

Context:
{context}
"""

def format_docs_with_metadata(docs):
    """Formats retrieved chunks with their clause headers for LLM consumption."""
    formatted = []
    for doc in docs:
        clause = doc.metadata.get("clause_id", "General Clause")
        page = doc.metadata.get("page", "Unknown")
        formatted.append(f"[{clause} | Page {page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(formatted)

class CEDDRAGPipeline:
    def __init__(self, k: int = 4):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in your .env file.")

        print("Initializing CEDD RAG Pipeline...")
        # 1. Load data and initialize retriever
        self.chunks = load_and_chunk_specification()
        self.vectorstore = build_vector_store(self.chunks, force_reindex=False)
        self.retriever = build_hybrid_retriever(self.chunks, self.vectorstore, k=k)

        # 2. Configure LLM via OpenRouter
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model="nvidia/nemotron-3.5-lightning:free",
            temperature=0.1,  # Low temperature for precise engineering extraction
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "CEDD Section 15 RAG"
            }
        )

        # 3. Build Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        # 4. Chain: format context -> prompt -> llm -> string output
        self.chain = (
            {
                "context": lambda x: format_docs_with_metadata(x["docs"]),
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def answer_query(self, question: str):
        """Retrieves documents, generates an answer, and returns both."""
        retrieved_docs = self.retriever.invoke(question)
        answer = self.chain.invoke({
            "docs": retrieved_docs,
            "question": question
        })
        return {
            "question": question,
            "answer": answer,
            "source_documents": retrieved_docs
        }

if __name__ == "__main__":
    rag = CEDDRAGPipeline(k=4)
    sample_question = "What are the temperature requirements when bending steel reinforcement?"
    
    print(f"\nAsking Question: {sample_question}\n")
    result = rag.answer_query(sample_question)
    
    print("=== Generated Answer ===")
    print(result["answer"])
    print("\n=== Sources Cited ===")
    for doc in result["source_documents"][:2]:
        print(f"- {doc.metadata.get('clause_id')} (Page {doc.metadata.get('page')})")