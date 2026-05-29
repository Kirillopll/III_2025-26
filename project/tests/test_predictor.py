from __future__ import annotations

from credit_risk_service.ml.predictor import risk_category


def test_risk_category_thresholds() -> None:
    assert risk_category(0.03, 0.10, 0.25) == "low"
    assert risk_category(0.12, 0.10, 0.25) == "medium"
    assert risk_category(0.40, 0.10, 0.25) == "high"
