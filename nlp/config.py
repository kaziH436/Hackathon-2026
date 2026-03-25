"""
config.py  — updated to include LangChain + SHAP settings
Merges into your existing config. Add/replace the fields below.
"""

from pydantic_settings import BaseSettings
from typing import List
import torch
from pathlib import Path

NLP_ROOT     = Path(__file__).resolve().parent        # .../Hackathon-2026/nlp
PROJECT_ROOT = NLP_ROOT.parent  

LABELS: List[str] = [
    "congestion_charge",
    "lane_closure",
    "public_transport_increase",
    "signal_retiming",
     "speed_limit_reduction"
]


MODEL_DIR      = PROJECT_ROOT / "models" / "policy_bert"
MAX_BATCH_SIZE = 32
MAX_LENGTH     = 128

class Settings(BaseSettings):

    # ── Existing BERT / API settings (keep yours) ─────────────────────────
    APP_NAME: str         = "City Policy NLP API"
    API_VERSION: str      = "v1"
    DEBUG: bool           = False
    DEVICE: str           = "cuda" if torch.cuda.is_available() else "cpu"

    
    # ── Your fine-tuned BERT model path ───────────────────────────────────
    # This folder should contain: config.json, tokenizer.json,
    # tokenizer_config.json, pytorch_model.bin (or model.safetensors)
    BERT_MODEL_PATH: str  = str(PROJECT_ROOT / "models" / "policy_bert")       # adjust to your actual path
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # ── NEW: Generator LLM for Q&A answers ───────────────────────────────
    # Flan-T5 generates the answer from retrieved context.
    # Swap for a larger model when you need richer answers.
    GENERATOR_MODEL: str  = "google/flan-t5-base"

    # ── NEW: Data directory (your text + CSV city reports) ───────────────
    DATA_DIR: str         = str(PROJECT_ROOT / "datasets")        # folder with .txt and .csv files

    # ── NEW: FAISS vector index path ─────────────────────────────────────
    FAISS_INDEX_PATH: str = "./faiss_index"

    # ── NEW: Retrieval settings ───────────────────────────────────────────
    TOP_K_DOCS: int       = 4              # how many chunks to retrieve per query

    # ── NEW: Policy class labels (must match your BERT fine-tuning labels) ─
    # Update this list to match the labels in your config.json id2label map

    POLICY_LABELS: List[str] = [
        "congestion_charge",
        "lane_closure",
        "public_transport_increase",
        "signal_retiming",
        "speed_limit_reduction"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
