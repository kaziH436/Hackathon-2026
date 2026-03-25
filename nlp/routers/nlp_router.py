"""
routers/nlp_router.py
New routes to plug into your existing main.py.

In main.py add:
    from routers.nlp_router import router as nlp_router
    app.include_router(nlp_router, prefix="/api/v1/nlp", tags=["NLP"])
"""

from fastapi import APIRouter, HTTPException, Depends
from schemas import (
    QARequest, QAResponse, QASource,
    ExplainPolicyRequest, ExplainPolicyResponse,
    ExplainQARequest,
    TokenImportanceRequest, TokenImportanceResponse,
)
from qa_chain import get_qa_chain, CityQAChain
from shap_explainer import get_shap_explainer, BertSHAPExplainer
import main

router = APIRouter()


# ── Dependency injection (FastAPI loads each singleton once) ──────────────
def qa_chain_dep() -> CityQAChain:
    return get_qa_chain()

def shap_dep() -> BertSHAPExplainer:
    return get_shap_explainer()


# ════════════════════════════════════════════════
# Q&A Endpoints
# ════════════════════════════════════════════════

@router.post("/qa", response_model=QAResponse, summary="Ask a question about city data")
async def query_city_data(
    body: QARequest,
    qa: CityQAChain = Depends(qa_chain_dep),
    shap: BertSHAPExplainer = Depends(shap_dep),
):
    """
    Ask a natural language question over city reports and simulation CSVs.
    Set explain=true to include token-level SHAP for the answer.
    """
    try:
        result = qa.ask(body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA chain error: {str(e)}")

    explanation = None
    if body.explain and result["raw_docs"]:
        # explain against the top retrieved chunk
        try:
            explanation = shap.explain_qa_answer(
                question=body.question,
                context=result["raw_docs"][0],
            )
        except Exception as e:
            explanation = {"error": str(e)}
    bert_label = main.classifier.predict(body.question)["label"]  
    
    return QAResponse(
        question=body.question,
        answer=result["answer"],
        sources=[QASource(**s) for s in result["sources"]],
        explanation=explanation,
        bert_label=bert_label,
    )


# ════════════════════════════════════════════════
# SHAP Explainability Endpoints
# ════════════════════════════════════════════════

@router.post(
    "/explain/policy",
    response_model=ExplainPolicyResponse,
    summary="Explain a BERT policy recommendation",
)
async def explain_policy(
    body: ExplainPolicyRequest,
    shap: BertSHAPExplainer = Depends(shap_dep),
):
    """
    Returns which tokens most influenced BERT's policy recommendation
    (APPROVE / REJECT / NEEDS_REVIEW) and by how much.
    """
    try:
        result = shap.explain_policy(body.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP error: {str(e)}")

    return ExplainPolicyResponse(
        prediction=result["prediction"],
        all_class_scores=result["all_class_scores"],
        top_tokens=result["top_tokens"],
        base_value=result["base_value"],
        explanation=result["explanation"],
    )


@router.post(
    "/explain/qa",
    summary="Explain Q&A answer confidence against a context chunk",
)
async def explain_qa(
    body: ExplainQARequest,
    shap: BertSHAPExplainer = Depends(shap_dep),
):
    """
    Shows which tokens in the retrieved context contributed to
    BERT's confidence in the Q&A answer.
    """
    try:
        return shap.explain_qa_answer(body.question, body.context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/explain/tokens",
    response_model=TokenImportanceResponse,
    summary="Raw token-level SHAP importance across all classes",
)
async def token_importance(
    body: TokenImportanceRequest,
    shap: BertSHAPExplainer = Depends(shap_dep),
):
    """
    Returns a heatmap-ready token→SHAP matrix across all policy classes.
    Useful for frontend visualizations.
    """
    try:
        tokens = shap.token_importance(body.text)
        return TokenImportanceResponse(tokens=tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check for NLP subsystem ───────────────────────────────────────
@router.get("/health", summary="NLP subsystem health check")
async def nlp_health():
    return {
        "qa_chain": "ready",
        "shap_explainer": "ready",
        "status": "ok",
    }
