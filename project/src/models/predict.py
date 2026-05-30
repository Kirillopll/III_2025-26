from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import load_config, resolve_path
from src.features.build_features import payload_to_frame


def model_path_from_config() -> Path:
    return resolve_path(load_config()["artifacts"]["model_path"])


@lru_cache(maxsize=1)
def load_model(path: str | Path | None = None) -> Any:
    model_path = resolve_path(path) if path else model_path_from_config()
    if not model_path.exists():
        raise FileNotFoundError(f"Артефакт модели не найден: {model_path}. Сначала запустите `python -m src.train`.")
    return joblib.load(model_path)


def risk_category(probability: float) -> str:
    service_cfg = load_config()["service"]
    if probability < float(service_cfg["low_risk_threshold"]):
        return "low"
    if probability < float(service_cfg["high_risk_threshold"]):
        return "medium"
    return "high"


def predict_one(payload: dict[str, Any]) -> dict[str, Any]:
    model = load_model()
    frame = payload_to_frame(payload)
    probability = float(model.predict_proba(frame)[0][1])
    return {
        "default_probability": probability,
        "risk_category": risk_category(probability),
        "model": load_config()["project"]["name"],
    }


def predict_batch(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model = load_model()
    frames = [payload_to_frame(record) for record in records]
    frame = pd.concat(frames, ignore_index=True)
    probabilities = model.predict_proba(frame)[:, 1]
    return [
        {
            "default_probability": float(probability),
            "risk_category": risk_category(float(probability)),
            "model": load_config()["project"]["name"],
        }
        for probability in probabilities
    ]
