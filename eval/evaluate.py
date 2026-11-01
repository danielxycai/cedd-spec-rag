import sys
import pandas as pd
from pathlib import Path

# Ensure workspace root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingest import load_and_chunk_specification
from src.indexer import build_vector_store, build_hybrid_retriever

# 1. Ground-Truth Dataset (Question -> Expected Clause ID)
GROUND_TRUTH = [
    {
        "question": "What are the temperature requirements when bending steel reinforcement?",
        "expected_clause": "Clause 15.20"
    },
    {
        "question": "How many days before fixing must particulars of Category A reinforcement be submitted?",
        "expected_clause": "Clause 15.12"
    },
    {
        "question": "What size tying wire should be used for stainless steel reinforcement?",
        "expected_clause": "Clause 15.10"
    },
    {
        "question": "How should epoxy-coated reinforcement be stored on site?",
        "expected_clause": "Clause 15.19"
    },
    {
        "question": "What is the acceptable tolerance for actual concrete cover to the outermost layer of reinforcement?",
        "expected_clause": "Clause 15.28"
    },
    {
        "question": "Under what condition is welding of reinforcement permitted?",
        "expected_clause": "Clause 15.25"
    },
    {
        "question": "How many specimens are required for testing a batch of less than 100 tension joint connectors?",
        "expected_clause": "Clause 15.31"
    }
]

def evaluate_retrieval(k: int = 4):
    """Evaluates the Hybrid Search pipeline against the ground truth dataset."""
    print("Loading chunks and initializing Hybrid Search Index...\n")
    chunks = load_and_chunk_specification()
    vectorstore = build_vector_store(chunks, force_reindex=False)
    retriever = build_hybrid_retriever(chunks, vectorstore, k=k)

    hits = 0
    mrr_sum = 0.0
    results_log = []

    print(f"Evaluating {len(GROUND_TRUTH)} ground-truth questions (Retrieving top {k} chunks)...\n")

    for item in GROUND_TRUTH:
        question = item["question"]
        expected = item["expected_clause"]

        # 2. Retrieve documents
        retrieved_docs = retriever.invoke(question)
        
        # Extract clean list of retrieved clause IDs
        retrieved_clauses = [doc.metadata.get("clause_id", "Unknown") for doc in retrieved_docs]

        # 3. Calculate Hit and Rank
        hit = 0
        rank = 0
        for r_idx, doc in enumerate(retrieved_docs, 1):
            primary = doc.metadata.get("clause_id", "")
            all_c = doc.metadata.get("all_clauses", [])
            clause_num = expected.replace("Clause ", "").strip()
            
            # Match against primary tag, all contained clause tags, or raw text
            if (expected in primary) or (expected in all_c) or (f"{clause_num}" in doc.page_content):
                hit = 1
                rank = r_idx
                break

        hits += hit
        if hit:
            mrr_sum += (1.0 / rank)

        # Log for display
        results_log.append({
            "Question (Truncated)": question[:50] + "...",
            "Expected": expected,
            "Hit": "✅ Yes" if hit else "❌ No",
            "Rank": rank if hit else "-",
            "Top Retrieved": retrieved_clauses[0] if retrieved_clauses else "None"
        })

    # 4. Compute Final Metrics
    hit_rate = hits / len(GROUND_TRUTH)
    mrr = mrr_sum / len(GROUND_TRUTH)

    # 5. Print Results
    df = pd.DataFrame(results_log)
    print(df.to_markdown(index=False))
    
    print("\n" + "="*40)
    print(f"🏆 RETRIEVAL EVALUATION METRICS (@{k})")
    print("="*40)
    print(f"Hit Rate : {hit_rate:.2%} (Found anywhere in top {k})")
    print(f"MRR      : {mrr:.4f} (Average rank position)")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_retrieval(k=4)