"""
nlp/data_loader.py
Loads your text files and CSVs into LangChain Document objects,
splits them into chunks, and feeds them into the FAISS index builder.

Run this ONCE as a script to build your FAISS index:
    python -m nlp.data_loader
"""

# REPLACE with these
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import glob
import os

from config import settings


# ── 1. Load Text Files ────────────────────────────────────────────────────
def load_text_files(data_dir: str) -> list[Document]:
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    return loader.load()


# ── 2. Load CSV Files ─────────────────────────────────────────────────────
def load_csv_files(data_dir: str) -> list[Document]:
    """
    CSVLoader treats each ROW as a separate Document.
    Each row becomes searchable independently — good for simulation data.
    """
    docs = []
    for csv_path in glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True):
        loader = CSVLoader(
            file_path=csv_path,
            csv_args={"delimiter": ","},
            source_column=None,     # set to a column name if you want it as source
        )
        loaded = loader.load()
        # tag each doc with its source file
        for doc in loaded:
            doc.metadata["source"] = os.path.basename(csv_path)
        docs.extend(loaded)
    return docs


# ── 3. Chunk Documents ────────────────────────────────────────────────────
def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Splits large documents into overlapping chunks so BERT
    embeddings stay within the 512-token limit.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,         # ~400 chars ≈ ~100 tokens, safe for BERT
        chunk_overlap=50,       # overlap keeps context across chunk boundaries
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# ── 4. Full Pipeline ──────────────────────────────────────────────────────
def load_all_documents(data_dir: str = None) -> list[Document]:
    data_dir = data_dir or settings.DATA_DIR

    print(f"[DataLoader] Loading from: {data_dir}")
    text_docs = load_text_files(data_dir)
    csv_docs  = load_csv_files(data_dir)

    all_docs  = text_docs + csv_docs
    print(f"[DataLoader] Raw docs: {len(all_docs)} (text={len(text_docs)}, csv={len(csv_docs)})")

    chunked = chunk_documents(all_docs)
    print(f"[DataLoader] Chunked into {len(chunked)} segments")
    return chunked


# ── 5. Run Once to Build FAISS Index ─────────────────────────────────────
if __name__ == "__main__":
    from qa_chain import load_embeddings
    from langchain_community.vectorstores import FAISS

    docs = load_all_documents()
    embeddings = load_embeddings()

    print("[DataLoader] Building FAISS index …")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(settings.FAISS_INDEX_PATH)
    print(f"[DataLoader] ✅ Index saved to {settings.FAISS_INDEX_PATH}")
