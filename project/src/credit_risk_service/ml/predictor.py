from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    default_probability: float
    risk_category: str


def risk_category(probability: float, low_threshold: float = 0.10, high_threshold: float = 0.25) -> str:
    if probability < low_threshold:
        return "low"
    if probability < high_threshold:
        return "medium"
    return "high"


class CreditRiskPredictor:
    def __init__(
        self,
        model_path: Path | str,
        low_risk_threshold: float = 0.10,
        high_risk_threshold: float = 0.25,
    ) -> None:
        self.model_path = Path(model_path)
        self.low_risk_threshold = low_risk_threshold
        self.high_risk_threshold = high_risk_threshold
        self.artifact: dict[str, Any] | None = None

    @property
    def is_ready(self) -> bool:
        return self.artifact is not None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Артефакт модели не найден: {self.model_path}. Сначала запустите `python -m credit_risk_service.ml.train`."
            )
        self.artifact = joblib.load(self.model_path)

    def predict_one(self, features: dict[str, Any]) -> PredictionResult:
        if self.artifact is None:
            self.load()

        assert self.artifact is not None
        model = self.artifact["model"]
        feature_columns = self.artifact["feature_columns"]
        row = pd.DataFrame([{column: features.get(column) for column in feature_columns}])

        probability = float(model.predict_proba(row)[0][1])
        return PredictionResult(
            default_probability=probability,
            risk_category=risk_category(probability, self.low_risk_threshold, self.high_risk_threshold),
        )

    def model_info(self) -> dict[str, Any]:
        if self.artifact is None:
            self.load()
        assert self.artifact is not None
        return {
            "model_name": self.artifact.get("model_name"),
            "trained_at": self.artifact.get("trained_at"),
            "metrics": self.artifact.get("metrics", {}),
            "feature_count": len(self.artifact.get("feature_columns", [])),
            "target_column": self.artifact.get("target_column"),
        }
