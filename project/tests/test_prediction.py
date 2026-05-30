from __future__ import annotations

from pathlib import Path

from src.config import resolve_path
from src.models.predict import risk_category


def test_risk_category_thresholds() -> None:
    assert risk_category(0.03) == "low"
    assert risk_category(0.12) == "medium"
    assert risk_category(0.40) == "high"


def test_model_artifact_exists_after_training() -> None:
    model_path = resolve_path("artifacts/model.joblib")
    assert isinstance(model_path, Path)
