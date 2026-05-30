from __future__ import annotations

from fastapi.testclient import TestClient

from src.service.app import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "model_missing"}
    assert "version" in body


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "requests_total" in body
    assert "errors_total" in body


def test_ui_page_is_available() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Оценка кредитного риска" in response.text
    assert "Рассчитать риск" in response.text
    assert "Дата рождения" in response.text


def test_predict_endpoint(sample_payload: dict) -> None:
    response = client.post("/predict", json=sample_payload)
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        body = response.json()
        assert 0 <= body["default_probability"] <= 1
        assert body["risk_category"] in {"low", "medium", "high"}
