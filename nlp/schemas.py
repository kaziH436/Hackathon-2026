"""
schemas.py — updated with Q&A and SHAP schemas
Add these to your existing schemas.py
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class PredictRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=512)

class PredictResponse(BaseModel):
    query:          str
    label:          str
    confidence:     float
    probabilities:  Dict[str, float]
    latency_ms:     float
    low_confidence: bool = False

class BatchPredictRequest(BaseModel):
    queries: List[str]

class BatchPredictResponse(BaseModel):
    predictions:      List[PredictResponse]
    total_latency_ms: float

class HealthResponse(BaseModel):
    status:    str
    model_dir: str
    labels:    List[str]
    device:    str

class LabelsResponse(BaseModel):
    labels: Dict[str, int]
# ════════════════════════════════════════════════
# Q&A Layer Schemas
# ════════════════════════════════════════════════




class QARequest(BaseModel):
    question: str = Field(
        ...,
        example="What are the traffic impact projections for Zone 4?",
        description="Natural language question about city reports or simulations",
    )
    explain: bool = Field(
        default=False,
        description="If true, also returns SHAP token-level explanation for the answer",
    )


class QASource(BaseModel):
    source: str          # filename the chunk came from
    snippet: str         # first 200 chars of the retrieved chunk


class QAResponse(BaseModel):
    question: str
    answer: str
    sources: List[QASource]
    explanation: Optional[dict] = None   # populated when explain=True


# ════════════════════════════════════════════════
# SHAP Explainability Schemas
# ════════════════════════════════════════════════

class ExplainPolicyRequest(BaseModel):
    text: str = Field(
        ...,
        example="Approve new zoning for residential development in sector 7.",
        description="Policy text to explain",
    )


class TokenSHAP(BaseModel):
    token:      str
    shap_value: float


class PolicyPrediction(BaseModel):
    label:      str
    confidence: float


class ExplainPolicyResponse(BaseModel):
    prediction:        PolicyPrediction
    all_class_scores:  Dict[str, float]
    top_tokens:        List[TokenSHAP]    # top 15 most influential tokens
    base_value:        float
    explanation:       str                # human-readable summary


class ExplainQARequest(BaseModel):
    question: str
    context:  str = Field(
        ...,
        description="Retrieved context snippet the answer was drawn from",
    )


class TokenImportanceRequest(BaseModel):
    text: str


class TokenImportanceResponse(BaseModel):
    tokens: List[Dict]   # [{token, importance_per_class: {label: shap_val}}]
