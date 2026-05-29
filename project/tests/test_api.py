from __future__ import annotations

from credit_risk_service.api.main import create_app
from credit_risk_service.ml.predictor import PredictionResult


class FakePredictor:
    model_path = "fake-model.joblib"

    def __init__(self) -> None:
        self.loaded = False

    @property
    def is_ready(self) -> bool:
        return self.loaded

    def load(self) -> None:
        self.loaded = True

    def predict_one(self, features: dict) -> PredictionResult:
        assert "AMT_INCOME_TOTAL" in features
        return PredictionResult(default_probability=0.18, risk_category="medium")

    def model_info(self) -> dict:
        return {
            "model_name": "fake",
            "trained_at": "2026-05-29T00:00:00+00:00",
            "metrics": {"average_precision": 0.5},
            "feature_count": 3,
            "target_column": "TARGET",
        }


def test_health_and_predict() -> None:
    from fastapi.testclient import TestClient

    app = create_app(predictor=FakePredictor())
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model_loaded": True}

        response = client.post("/predict", json={"features": {"AMT_INCOME_TOTAL": 150000}})
        assert response.status_code == 200
        assert response.json()["risk_category"] == "medium"
        assert 0 <= response.json()["default_probability"] <= 1


def test_model_info() -> None:
    from fastapi.testclient import TestClient

    app = create_app(predictor=FakePredictor())
    with TestClient(app) as client:
        response = client.get("/model-info")
        assert response.status_code == 200
        assert response.json()["model_name"] == "fake"
