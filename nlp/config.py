"""
nlp/config.py
Single source of truth for paths, label vocabulary, and inference constants.
Must match the label order used during BERT fine-tuning (Cell 4 of the notebook).
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Assumes the project root contains: nlp/ and models/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR     = _PROJECT_ROOT / "models" / "policy_bert"

# ── Label vocabulary ──────────────────────────────────────────────────────────
# Order MUST match LABELS list in Cell 4 of train_bert_colab.ipynb.
# BERT's output logit index 0 = congestion_charge, index 4 = speed_limit_reduction.
LABELS = [
    "congestion_charge",
    "lane_closure",
    "public_transport_increase",
    "signal_retiming",
    "speed_limit_reduction",
]

# ── Inference constants ───────────────────────────────────────────────────────
MAX_LENGTH    = 128   # must match MAX_LENGTH used during training
MAX_BATCH_SIZE = 32   # maximum queries per /predict/batch request

# ── Confidence threshold ──────────────────────────────────────────────────────
# Predictions below this threshold are flagged as low-confidence.
# Useful for the LangChain layer to decide whether to ask for clarification.
LOW_CONFIDENCE_THRESHOLD = 0.80