from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from credit_risk_service.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from credit_risk_service.core.config import get_service_settings
from credit_risk_service.core.logging import configure_logging
from credit_risk_service.ml.predictor import CreditRiskPredictor


REQUEST_COUNTER = Counter("credit_risk_requests_total", "Общее количество запросов к API", ["endpoint", "status"])
PREDICTION_LATENCY = Histogram("credit_risk_prediction_seconds", "Время ответа модели в секундах")

logger = logging.getLogger(__name__)


def create_app(predictor: CreditRiskPredictor | None = None) -> FastAPI:
    settings = get_service_settings()
    configure_logging(settings.log_level)
    selected_predictor = predictor or CreditRiskPredictor(
        model_path=settings.model_path,
        low_risk_threshold=settings.low_risk_threshold,
        high_risk_threshold=settings.high_risk_threshold,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = selected_predictor
        try:
            app.state.predictor.load()
            logger.info("Модель загружена из %s", app.state.predictor.model_path)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
        yield

    app = FastAPI(
        title="Сервис оценки кредитного риска",
        description="Оценивает кредитную заявку и возвращает вероятность дефолта вместе с категорией риска.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        ready = app.state.predictor.is_ready
        REQUEST_COUNTER.labels(endpoint="/health", status="ok").inc()
        return HealthResponse(status="ok" if ready else "degraded", model_loaded=ready)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        try:
            info = app.state.predictor.model_info()
            REQUEST_COUNTER.labels(endpoint="/model-info", status="ok").inc()
            return ModelInfoResponse(**info)
        except FileNotFoundError as exc:
            REQUEST_COUNTER.labels(endpoint="/model-info", status="error").inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        started_at = time.perf_counter()
        try:
            result = app.state.predictor.predict_one(request.features)
        except FileNotFoundError as exc:
            REQUEST_COUNTER.labels(endpoint="/predict", status="error").inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Prediction failed")
            REQUEST_COUNTER.labels(endpoint="/predict", status="error").inc()
            raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

        PREDICTION_LATENCY.observe(time.perf_counter() - started_at)
        REQUEST_COUNTER.labels(endpoint="/predict", status="ok").inc()
        return PredictionResponse(
            default_probability=result.default_probability,
            risk_category=result.risk_category,
        )

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
