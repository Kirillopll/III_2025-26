from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import load_config


@dataclass(frozen=True)
class FeatureSpec:
    categorical: list[str]
    numeric: list[str]
    target: str

    @property
    def all_features(self) -> list[str]:
        return self.categorical + self.numeric


def get_feature_spec() -> FeatureSpec:
    feature_cfg = load_config()["features"]
    return FeatureSpec(
        categorical=list(feature_cfg["categorical"]),
        numeric=list(feature_cfg["numeric"]),
        target=str(feature_cfg["target"]),
    )


def prepare_supervised_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    spec = get_feature_spec()
    required = spec.all_features + [spec.target]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"В данных отсутствуют обязательные колонки: {missing}")

    frame = data[required].copy()
    frame = frame.dropna(subset=[spec.target])
    X = frame[spec.all_features]
    y = frame[spec.target].astype(int)
    return X, y


def payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    spec = get_feature_spec()
    missing = [column for column in spec.all_features if column not in payload]
    if missing:
        raise ValueError(f"В запросе отсутствуют обязательные поля: {missing}")
    return pd.DataFrame([{column: payload[column] for column in spec.all_features}])
