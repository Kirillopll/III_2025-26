from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


@dataclass(frozen=True)
class ServiceSettings:
    model_path: Path
    low_risk_threshold: float
    high_risk_threshold: float
    log_level: str


def load_yaml_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def get_service_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> ServiceSettings:
    config = load_yaml_config(config_path)
    service_config = config.get("service", {})

    model_path = os.getenv("MODEL_PATH", service_config.get("model_path", "models/credit_risk_model.joblib"))
    low_threshold = float(os.getenv("LOW_RISK_THRESHOLD", service_config.get("low_risk_threshold", 0.10)))
    high_threshold = float(os.getenv("HIGH_RISK_THRESHOLD", service_config.get("high_risk_threshold", 0.25)))

    if low_threshold >= high_threshold:
        raise ValueError("LOW_RISK_THRESHOLD must be lower than HIGH_RISK_THRESHOLD")

    return ServiceSettings(
        model_path=resolve_project_path(model_path),
        low_risk_threshold=low_threshold,
        high_risk_threshold=high_threshold,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
