"""
nlp/model.py
Loads the fine-tuned BERT checkpoint and runs classification inference.

Design decisions:
  - Model and tokenizer loaded once at startup (via PolicyClassifier.__init__)
  - Inference always runs under torch.no_grad() — no gradient computation
  - Device selection: CUDA if available, else CPU
  - Tokenizer: AutoTokenizer (loads BertTokenizerFast from tokenizer.json)
  - Single predict() and batch predict_batch() methods — batch is more
    efficient per query when multiple queries arrive together
  - Softmax applied here (not in training) — raw logits → probabilities
  - Returns confidence (max prob) and full probability distribution
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, BertForSequenceClassification

from nlp.config import LABELS, MAX_LENGTH


class PolicyClassifier:
    """
    Wraps the fine-tuned BertForSequenceClassification model.
    Thread-safe for read-only inference (FastAPI runs async but torch
    inference with no_grad is safe to call from multiple coroutines).
    """

    def __init__(self, model_dir: Path) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}\n"
                "Download policy_bert.zip from Colab Cell 11 and unzip it into "
                "models/policy_bert/ at the project root."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = str(self.device)

        # AutoTokenizer reads tokenizer.json — works with BertTokenizerFast
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        self.model = BertForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()  # disable dropout for inference

        # Build label lookup from model config (set during training in Cell 4)
        # Falls back to LABELS from config.py if not in model config
        if hasattr(self.model.config, "id2label"):
            self.id2label = self.model.config.id2label
        else:
            self.id2label = {idx: label for idx, label in enumerate(LABELS)}

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a list of strings and move tensors to device."""
        encoding = self.tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding=True,        # pad to longest in batch, not to MAX_LENGTH
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def _logits_to_result(
        self, logits: torch.Tensor, query: str
    ) -> dict:
        """Convert a single row of logits to a prediction dict."""
        probs = F.softmax(logits, dim=-1)  # softmax NOT applied during training
        confidence, pred_id = probs.max(dim=-1)

        label = self.id2label[pred_id.item()]
        conf  = round(confidence.item(), 6)

        probabilities = {
            self.id2label[i]: round(probs[i].item(), 6)
            for i in range(len(self.id2label))
        }

        return {
            "label":         label,
            "confidence":    conf,
            "probabilities": probabilities,
        }

    def predict(self, query: str) -> dict:
        """
        Classify a single query string.

        Returns:
            {
              "label":         str,          # predicted intervention label
              "confidence":    float,        # softmax probability of top label
              "probabilities": dict[str, float]  # full distribution
            }
        """
        return self.predict_batch([query])[0]

    def predict_batch(self, queries: list[str]) -> list[dict]:
        """
        Classify a batch of query strings in a single forward pass.
        More efficient than calling predict() in a loop.

        Returns:
            List of dicts, one per query, same order as input.
        """
        encoding = self._tokenize(queries)

        with torch.no_grad():
            outputs = self.model(**encoding)

        # outputs.logits shape: (batch_size, num_labels)
        results = []
        for i, query in enumerate(queries):
            result = self._logits_to_result(outputs.logits[i], query)
            results.append(result)

        return results