"""
nlp/schemas.py
Pydantic request and response models for the FastAPI endpoint.
Validation happens automatically before any route handler runs.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Annotated


# ── Shared types ──────────────────────────────────────────────────────────────

QueryStr = Annotated[
    str,
    Field(
        min_length=3,
        max_length=512,
        description="Natural language operator query to classify.",
        examples=["what happens to air quality on Wilshire if we cap speeds at 20mph?"],
    ),
]


# ── Request models ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    query: QueryStr

    @field_validator("query")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty or whitespace only.")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "what happens to air quality on Wilshire if we cap speeds at 20mph?"}
            ]
        }
    }


class BatchPredictRequest(BaseModel):
    queries: list[QueryStr] = Field(
        min_length=1,
        description="List of queries to classify in a single batch (max 32).",
    )

    @field_validator("queries")
    @classmethod
    def strip_each(cls, v: list[str]) -> list[str]:
        return [q.strip() for q in v if q.strip()]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "queries": [
                        "what if we charge drivers to enter downtown during rush hour?",
                        "close the northbound lane on the 405 for maintenance on Saturday",
                        "double bus frequency on Vermont Ave and show emission change",
                    ]
                }
            ]
        }
    }


# ── Response models ───────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    query: str = Field(description="The original query string.")
    label: str = Field(description="Predicted policy intervention label.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Softmax probability of the predicted label (0–1).",
    )
    probabilities: dict[str, float] = Field(
        description="Full softmax distribution across all five labels."
    )
    latency_ms: float = Field(description="Inference time in milliseconds.")
    low_confidence: bool = Field(
        default=False,
        description=(
            "True if confidence is below the LOW_CONFIDENCE_THRESHOLD (0.80). "
            "The LangChain layer can use this to ask the operator for clarification."
        ),
    )

    def model_post_init(self, __context) -> None:
        from nlp.config import LOW_CONFIDENCE_THRESHOLD
        # Set low_confidence flag after all fields are initialised
        object.__setattr__(self, "low_confidence", self.confidence < LOW_CONFIDENCE_THRESHOLD)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse] = Field(
        description="One prediction per query, in the same order as the request."
    )
    total_latency_ms: float = Field(
        description="Total wall-clock time for the entire batch in milliseconds."
    )


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' if model is loaded and ready.")
    model_dir: str = Field(description="Path to the loaded model checkpoint.")
    labels: list[str] = Field(description="Label vocabulary in ID order.")
    device: str = Field(description="Compute device: 'cuda' or 'cpu'.")


class LabelsResponse(BaseModel):
    labels: dict[str, int] = Field(
        description="Label name → integer ID mapping (matches training config)."
    )