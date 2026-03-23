"""
nlp/main.py
FastAPI endpoint for the policy classification BERT model.

Routes:
  POST /predict        — classify a single query
  POST /predict/batch  — classify a list of queries (up to 32)
  GET  /health         — liveness check
  GET  /labels         — return the label vocabulary

Usage:
  uvicorn nlp.main:app --host 0.0.0.0 --port 8000 --reload

Expected project layout:
  nlp/
    main.py          ← this file
    model.py         ← model loading + inference
    schemas.py       ← Pydantic request/response models
    config.py        ← paths and constants
  models/
    policy_bert/     ← saved BERT checkpoint (from Colab Cell 11)
      config.json
      model.safetensors
      tokenizer.json
      tokenizer_config.json
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time

from nlp.model import PolicyClassifier
from nlp.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    LabelsResponse,
)
from nlp.config import MODEL_DIR, LABELS, MAX_BATCH_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global model instance ─────────────────────────────────────────────────────
# Loaded once at startup, shared across all requests.
classifier: PolicyClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global classifier
    logger.info(f"Loading policy classifier from {MODEL_DIR}...")
    t0 = time.perf_counter()
    classifier = PolicyClassifier(MODEL_DIR)
    elapsed = time.perf_counter() - t0
    logger.info(f"Model loaded in {elapsed:.2f}s — ready to serve.")
    yield
    # Cleanup on shutdown
    classifier = None
    logger.info("Model released.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart City NLP — Policy Classifier",
    description=(
        "Classifies natural language operator queries into one of five "
        "traffic/transport policy intervention categories using fine-tuned BERT."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the digital twin frontend to call this directly if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production to specific origins
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Liveness check. Returns 200 if the model is loaded and ready."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return HealthResponse(
        status="ok",
        model_dir=str(MODEL_DIR),
        labels=LABELS,
        device=classifier.device_name,
    )


@app.get("/labels", response_model=LabelsResponse, tags=["Meta"])
def labels() -> LabelsResponse:
    """Return the full label vocabulary with integer IDs."""
    return LabelsResponse(
        labels={label: idx for idx, label in enumerate(LABELS)}
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Classify a single natural language query.

    Returns the predicted policy intervention label, confidence score,
    and the full probability distribution across all five labels.

    Example request:
        {"query": "what happens to air quality on Wilshire if we cap speeds at 20mph?"}

    Example response:
        {
          "query": "what happens to air quality on Wilshire if we cap speeds at 20mph?",
          "label": "speed_limit_reduction",
          "confidence": 0.9987,
          "probabilities": {
            "congestion_charge": 0.0003,
            "lane_closure": 0.0001,
            "public_transport_increase": 0.0001,
            "signal_retiming": 0.0008,
            "speed_limit_reduction": 0.9987
          },
          "latency_ms": 18.4
        }
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()
    result = classifier.predict(request.query)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"predict | label={result['label']} conf={result['confidence']:.4f} "
        f"latency={latency_ms:.1f}ms | query={request.query[:80]}"
    )

    return PredictResponse(
        query=request.query,
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Classify a batch of queries in a single forward pass.

    More efficient than calling /predict repeatedly when you have
    multiple queries to classify at once (e.g. from a report parser).
    Maximum batch size: 32 queries per request.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(request.queries) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Batch size {len(request.queries)} exceeds maximum of "
                f"{MAX_BATCH_SIZE}. Split into smaller batches."
            ),
        )

    if not request.queries:
        raise HTTPException(status_code=422, detail="queries list is empty.")

    t0 = time.perf_counter()
    results = classifier.predict_batch(request.queries)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"predict_batch | n={len(request.queries)} "
        f"latency={latency_ms:.1f}ms"
    )

    predictions = [
        PredictResponse(
            query=q,
            label=r["label"],
            confidence=r["confidence"],
            probabilities=r["probabilities"],
            latency_ms=round(latency_ms / len(request.queries), 2),
        )
        for q, r in zip(request.queries, results)
    ]

    return BatchPredictResponse(
        predictions=predictions,
        total_latency_ms=round(latency_ms, 2),
    )