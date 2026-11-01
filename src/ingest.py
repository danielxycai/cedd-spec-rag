import os
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

PDF_PATH = Path("data/section_15_steel_reinforcement.pdf")

def clean_page_text(text: str) -> str:
    """Removes running headers, footers, and standalone page numbers (e.g., 15.5, 15.7)."""
    cleaned_lines = []
    for line in text.splitlines():
        line_strip = line.strip()
        # Filter out running document headers
        if re.match(r"^GS\s*\(2020\s*Edition\)$", line_strip, re.IGNORECASE):
            continue
        if re.match(r"^SECTION\s*15$", line_strip, re.IGNORECASE):
            continue
        if re.match(r"^STEEL\s*REINFORCEMENT$", line_strip, re.IGNORECASE):
            continue
        # Filter out standalone page numbers (e.g., "15.3", "15.7", "15.15" when alone on a line)
        if re.match(r"^15\.\d{1,2}$", line_strip):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def load_and_chunk_specification(pdf_path: Path = PDF_PATH):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}. Please place it in data/")

    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"Loaded {len(pages)} pages.")

    # 1. Clean each page and combine into a unified document text
    full_text = ""
    for page in pages:
        cleaned = clean_page_text(page.page_content)
        full_text += "\n" + cleaned

    # 2. Split continuously across page breaks by clause boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=200,
        separators=[
            r"\n(?=15\.\d{2}\b)",  # Strict regex for 2-digit clause numbers (15.01 - 15.40)
            "\n\n",
            "\n",
            " "
        ],
        is_separator_regex=True
    )

    raw_chunks = splitter.split_text(full_text)
    
    # 3. Track active clause and attach metadata
    documents = []
    current_clause = "Clause 15.01"

    for i, chunk_text in enumerate(raw_chunks):
        # Detect all clauses present inside this chunk
        found_clauses = re.findall(r"\b(15\.\d{2}[A-Z]?)\b", chunk_text)
        clause_list = [f"Clause {c}" for c in dict.fromkeys(found_clauses)]
        
        if clause_list:
            current_clause = clause_list[0]

        doc = Document(
            page_content=chunk_text.strip(),
            metadata={
                "clause_id": current_clause,
                "all_clauses": clause_list if clause_list else [current_clause],
                "chunk_index": i,
                "source": "CEDD GS Section 15"
            }
        )
        documents.append(doc)

    print(f"Successfully generated {len(documents)} clean chunks.")
    return documents

if __name__ == "__main__":
    sample_chunks = load_and_chunk_specification()
    print("\n--- Verification: Checking Clause 15.19 & Storage ---")
    for doc in sample_chunks:
        if "15.19" in doc.metadata["clause_id"] or "stored" in doc.page_content.lower():
            print(f"\nMetadata: {doc.metadata}")
            print(f"Snippet:\n{doc.page_content[:250]}...\n")
            break