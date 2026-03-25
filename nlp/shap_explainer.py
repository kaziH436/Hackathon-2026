"""
nlp/shap_explainer.py
SHAP explainability layer for your fine-tuned BERT.

Covers all three targets:
  1. Policy recommendation confidence  (classification output)
  2. Q&A answer confidence             (spans over retrieved context)
  3. Token-level importance            (which tokens drove the prediction)
"""

import shap
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from typing import Any

from config import settings


# ── 1. Load BERT Pipeline ─────────────────────────────────────────────────
def _load_bert_pipeline():
    """
    Wraps your fine-tuned BERT as a HuggingFace pipeline so
    SHAP's Explainer can call it like a black-box function.
    """
    return pipeline(
        "text-classification",
        model=settings.BERT_MODEL_PATH,
        tokenizer=settings.BERT_MODEL_PATH,
        device=0 if settings.DEVICE == "cuda" else -1,
        top_k=None,          # ← replaces return_all_scores=True
        truncation=True,
        max_length=512,
    )


# ── 2. SHAP Explainer (Text-based) ────────────────────────────────────────
class BertSHAPExplainer:
    """
    Singleton wrapper around shap.Explainer for BERT.
    Handles all three explanation modes.
    """

    def __init__(self):
        print("[SHAP] Loading BERT pipeline for explainability …")
        self.pipe = _load_bert_pipeline()
        self.tokenizer = AutoTokenizer.from_pretrained(settings.BERT_MODEL_PATH)

        # Use masker instead of tokenizer directly
        masker = shap.maskers.Text(self.tokenizer)

        self.explainer = shap.Explainer(
            self._predict_proba,
            masker,                          # ← pass masker not tokenizer
            output_names=settings.POLICY_LABELS,
        )
        print("[SHAP] ✅ Explainer ready")

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        # Ensure inputs are always plain strings
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]    # ← force everything to str
        
        results = self.pipe(texts, batch_size=8)
        
        # Handle single result vs batch
        if isinstance(results[0], dict):
            results = [results]
        
        probs = np.array([
            [score["score"] for score in sorted(r, key=lambda x: x["label"])]
            for r in results
        ])
        return probs

    # ── 2a. Policy Recommendation Explanation ────────────────────────────
    def explain_policy(self, text: str) -> dict:
        """
        Explains which tokens pushed BERT toward a policy recommendation.

        Returns:
          prediction     – top predicted label + confidence
          token_shap     – list of {token, shap_value} for the top class
          base_value     – SHAP base (expected model output)
        """
        shap_values = self.explainer([text])

        # index of highest-confidence class
        pred_scores  = self._predict_proba([text])[0]
        top_class_idx = int(np.argmax(pred_scores))
        top_label     = settings.POLICY_LABELS[top_class_idx]

        # token-level SHAP for the predicted class
        tokens      = shap_values.data[0]
        values      = shap_values.values[0][:, top_class_idx]

        token_shap = [
            {"token": tok, "shap_value": round(float(val), 4)}
            for tok, val in zip(tokens, values)
        ]
        token_shap_sorted = sorted(token_shap, key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "prediction": {
                "label":      top_label,
                "confidence": round(float(pred_scores[top_class_idx]), 4),
            },
            "all_class_scores": {
                label: round(float(score), 4)
                for label, score in zip(settings.POLICY_LABELS, pred_scores)
            },
            "top_tokens":   token_shap_sorted[:15],     # top 15 most influential
            "all_tokens":   token_shap,
            "base_value":   round(float(shap_values.base_values[0][top_class_idx]), 4),
            "explanation":  _build_explanation_text(token_shap_sorted[:5], top_label),
        }

    # ── 2b. Q&A Answer Confidence Explanation ────────────────────────────
    def explain_qa_answer(self, question: str, context: str) -> dict:
        """
        Explains how confident BERT is about the answer given the
        retrieved context, and which context tokens matter most.

        We format as a single text so SHAP can mask across both.
        """
        combined = f"Question: {question} Context: {context}"
        return self.explain_policy(combined)   # same SHAP flow, different input

    # ── 2c. Token-Level Importance (raw) ─────────────────────────────────
    def token_importance(self, text: str) -> list[dict]:
        """
        Returns raw token→SHAP importance for ALL classes.
        Useful for a heatmap visualization in the frontend.
        """
        shap_values = self.explainer([text])
        tokens = shap_values.data[0]
        values = shap_values.values[0]   # shape: (num_tokens, num_classes)

        return [
            {
                "token": tok,
                "importance_per_class": {
                    label: round(float(values[i][j]), 4)
                    for j, label in enumerate(settings.POLICY_LABELS)
                },
            }
            for i, tok in enumerate(tokens)
        ]


# ── Helper ────────────────────────────────────────────────────────────────
def _build_explanation_text(top_tokens: list[dict], label: str) -> str:
    positive = [t["token"] for t in top_tokens if t["shap_value"] > 0]
    negative = [t["token"] for t in top_tokens if t["shap_value"] < 0]

    parts = [f"Model predicted: '{label}'."]
    if positive:
        parts.append(f"Tokens supporting this: {', '.join(positive[:3])}.")
    if negative:
        parts.append(f"Tokens opposing this: {', '.join(negative[:3])}.")
    return " ".join(parts)


# ── Singleton ─────────────────────────────────────────────────────────────
_explainer_instance: BertSHAPExplainer | None = None


def get_shap_explainer() -> BertSHAPExplainer:
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = BertSHAPExplainer()
    return _explainer_instance
