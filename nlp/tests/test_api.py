"""
nlp/tests/test_api.py
Test suite for the FastAPI policy classifier endpoint.

Uses FastAPI's TestClient (backed by httpx) so no live server is needed.
The model is mocked so tests run without the actual BERT checkpoint —
useful in CI where the 440MB model file is not available.

Run:
    pytest nlp/tests/test_api.py -v

To run against the real model (integration tests):
    pytest nlp/tests/test_api.py -v --real-model
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_mock_classifier(label="congestion_charge", confidence=0.9987):
    """Return a mock PolicyClassifier that returns a fixed prediction."""
    mock = MagicMock()
    mock.device_name = "cpu"

    def _predict(query: str) -> dict:
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                "congestion_charge":        confidence if label == "congestion_charge" else 0.0001,
                "lane_closure":             0.0001,
                "public_transport_increase": 0.0001,
                "signal_retiming":          0.0001,
                "speed_limit_reduction":    0.0001,
            },
        }

    def _predict_batch(queries: list) -> list:
        return [_predict(q) for q in queries]

    mock.predict.side_effect = _predict
    mock.predict_batch.side_effect = _predict_batch
    return mock


@pytest.fixture
def client():
    """TestClient with mocked classifier — no model file needed."""
    mock_classifier = _make_mock_classifier()
    with patch("nlp.main.classifier", mock_classifier):
        from nlp.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def low_confidence_client():
    """TestClient with a low-confidence mock classifier."""
    mock_classifier = _make_mock_classifier(
        label="congestion_charge", confidence=0.62
    )
    with patch("nlp.main.classifier", mock_classifier):
        from nlp.main import app
        with TestClient(app) as c:
            yield c


# ── Health route ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_shape(self, client):
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert "model_dir" in body
        assert "labels" in body
        assert "device" in body

    def test_labels_present(self, client):
        r = client.get("/health")
        labels = r.json()["labels"]
        assert "congestion_charge" in labels
        assert "speed_limit_reduction" in labels
        assert len(labels) == 5


# ── Labels route ──────────────────────────────────────────────────────────────

class TestLabels:
    def test_returns_200(self, client):
        r = client.get("/labels")
        assert r.status_code == 200

    def test_five_labels(self, client):
        r = client.get("/labels")
        assert len(r.json()["labels"]) == 5

    def test_ids_are_integers(self, client):
        r = client.get("/labels")
        for label, idx in r.json()["labels"].items():
            assert isinstance(idx, int)


# ── Single predict route ──────────────────────────────────────────────────────

class TestPredict:
    def test_valid_query_returns_200(self, client):
        r = client.post("/predict", json={"query": "what if we add a toll on the 405?"})
        assert r.status_code == 200

    def test_response_has_required_fields(self, client):
        r = client.post("/predict", json={"query": "close the northbound lane on Wilshire"})
        body = r.json()
        assert "query" in body
        assert "label" in body
        assert "confidence" in body
        assert "probabilities" in body
        assert "latency_ms" in body
        assert "low_confidence" in body

    def test_confidence_is_between_0_and_1(self, client):
        r = client.post("/predict", json={"query": "simulate congestion pricing downtown"})
        body = r.json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, client):
        r = client.post("/predict", json={"query": "add more buses on Vermont Ave"})
        probs = r.json()["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.001

    def test_five_probabilities_returned(self, client):
        r = client.post("/predict", json={"query": "retime signals at Hollywood and Highland"})
        assert len(r.json()["probabilities"]) == 5

    def test_query_echoed_in_response(self, client):
        query = "what does a 20mph cap do to emissions near Koreatown?"
        r = client.post("/predict", json={"query": query})
        assert r.json()["query"] == query

    def test_low_confidence_flag_false_when_confident(self, client):
        r = client.post("/predict", json={"query": "toll on the 405 during rush hour"})
        assert r.json()["low_confidence"] is False

    def test_low_confidence_flag_true_when_uncertain(self, low_confidence_client):
        r = low_confidence_client.post("/predict", json={"query": "something ambiguous"})
        assert r.json()["low_confidence"] is True

    def test_latency_ms_is_positive(self, client):
        r = client.post("/predict", json={"query": "close the center lane on the 101"})
        assert r.json()["latency_ms"] > 0

    # ── Input validation ──────────────────────────────────────────────────────

    def test_empty_query_returns_422(self, client):
        r = client.post("/predict", json={"query": ""})
        assert r.status_code == 422

    def test_whitespace_only_query_returns_422(self, client):
        r = client.post("/predict", json={"query": "   "})
        assert r.status_code == 422

    def test_too_short_query_returns_422(self, client):
        r = client.post("/predict", json={"query": "hi"})
        assert r.status_code == 422

    def test_too_long_query_returns_422(self, client):
        r = client.post("/predict", json={"query": "x" * 513})
        assert r.status_code == 422

    def test_missing_query_field_returns_422(self, client):
        r = client.post("/predict", json={"text": "wrong field name"})
        assert r.status_code == 422

    def test_non_string_query_returns_422(self, client):
        r = client.post("/predict", json={"query": 12345})
        assert r.status_code == 422


# ── Batch predict route ───────────────────────────────────────────────────────

class TestPredictBatch:
    def test_valid_batch_returns_200(self, client):
        r = client.post("/predict/batch", json={
            "queries": [
                "toll on the 405 during rush hour",
                "close the northbound lane on Wilshire",
                "double bus frequency near Downtown LA",
            ]
        })
        assert r.status_code == 200

    def test_response_count_matches_request(self, client):
        queries = [
            "congestion charge in the financial district",
            "close the fast lane on the 101",
            "more buses on Vermont Ave",
        ]
        r = client.post("/predict/batch", json={"queries": queries})
        assert len(r.json()["predictions"]) == len(queries)

    def test_total_latency_present(self, client):
        r = client.post("/predict/batch", json={
            "queries": ["toll on Sunset Blvd during evening rush"]
        })
        assert "total_latency_ms" in r.json()
        assert r.json()["total_latency_ms"] > 0

    def test_single_item_batch(self, client):
        r = client.post("/predict/batch", json={
            "queries": ["reduce speed limit near the school zone"]
        })
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 1

    def test_empty_batch_returns_422(self, client):
        r = client.post("/predict/batch", json={"queries": []})
        assert r.status_code == 422

    def test_batch_too_large_returns_422(self, client):
        queries = [f"query number {i}" for i in range(33)]
        r = client.post("/predict/batch", json={"queries": queries})
        assert r.status_code == 422

    def test_each_prediction_has_required_fields(self, client):
        r = client.post("/predict/batch", json={
            "queries": ["toll on the 101", "close a lane on Wilshire"]
        })
        for pred in r.json()["predictions"]:
            assert "label" in pred
            assert "confidence" in pred
            assert "probabilities" in pred
            assert "low_confidence" in pred