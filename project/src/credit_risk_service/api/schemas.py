from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Признаки кредитной заявки. Ключи должны соответствовать колонкам application_train.csv без TARGET.",
    )


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., ge=0.0, le=1.0)
    risk_category: Literal["low", "medium", "high"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str | None
    trained_at: str | None
    metrics: dict[str, float]
    feature_count: int
    target_column: str | None
