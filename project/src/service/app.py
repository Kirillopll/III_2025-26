from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import __version__
from src.config import load_config, resolve_path
from src.models.predict import load_model, predict_batch, predict_one


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("credit_risk_service")

config = load_config()
artifact_cfg = config["artifacts"]


class PredictionRequest(BaseModel):
    NAME_CONTRACT_TYPE: str = "Cash loans"
    CODE_GENDER: str = "M"
    FLAG_OWN_CAR: str = "N"
    NAME_INCOME_TYPE: str = "Working"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"
    NAME_FAMILY_STATUS: str = "Single / not married"
    NAME_HOUSING_TYPE: str = "House / apartment"
    WEEKDAY_APPR_PROCESS_START: str = "MONDAY"
    ORGANIZATION_TYPE: str = "Business Entity Type 3"
    CNT_CHILDREN: int = 0
    AMT_INCOME_TOTAL: float = Field(202500, gt=0)
    AMT_CREDIT: float = Field(406597, gt=0)
    AMT_ANNUITY: float = Field(24700, gt=0)
    DAYS_BIRTH: int = -9461
    DAYS_EMPLOYED: int = -637
    HOUR_APPR_PROCESS_START: int = Field(10, ge=0, le=23)
    EXT_SOURCE_2: float | None = 0.26
    EXT_SOURCE_3: float | None = 0.14


class BatchPredictionRequest(BaseModel):
    records: list[PredictionRequest] = Field(..., max_length=100)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
        logger.info("Модель загружена")
    except FileNotFoundError as exc:
        logger.warning("%s", exc)
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Оценка вероятности дефолта клиента по признакам кредитной заявки.",
    version=__version__,
    lifespan=lifespan,
)
app.mount("/artifacts", StaticFiles(directory=resolve_path("artifacts")), name="artifacts")

service_metrics: dict[str, float | int] = {
    "requests_total": 0,
    "errors_total": 0,
    "prediction_requests_total": 0,
    "request_duration_seconds_sum": 0.0,
}


@app.middleware("http")
async def collect_metrics(request: Request, call_next):
    started_at = perf_counter()
    service_metrics["requests_total"] += 1
    if request.url.path.startswith("/predict"):
        service_metrics["prediction_requests_total"] += 1
    try:
        response = await call_next(request)
    except Exception:
        service_metrics["errors_total"] += 1
        raise
    duration = perf_counter() - started_at
    service_metrics["request_duration_seconds_sum"] += duration
    if response.status_code >= 500:
        service_metrics["errors_total"] += 1
    logger.info("%s %s -> %s %.4fs", request.method, request.url.path, response.status_code, duration)
    return response


UI_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Оценка кредитного риска</title>
  <style>
    body { margin: 0; font-family: Segoe UI, system-ui, sans-serif; background: #f5f7fb; color: #18212f; }
    header { background: #ffffff; border-bottom: 1px solid #d8e0ec; }
    .wrap { width: min(1100px, calc(100% - 32px)); margin: 0 auto; }
    .top { padding: 18px 0; display: flex; justify-content: space-between; align-items: center; gap: 16px; }
    h1 { margin: 0; font-size: 24px; }
    main { padding: 24px 0 40px; display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 20px; }
    section { background: #fff; border: 1px solid #d8e0ec; border-radius: 8px; padding: 20px; box-shadow: 0 12px 28px rgba(20, 31, 47, .08); }
    h2 { margin: 0 0 16px; font-size: 18px; }
    form { display: grid; gap: 14px; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    label { display: grid; gap: 6px; font-size: 13px; color: #5b6678; font-weight: 650; }
    input, select { min-height: 40px; border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 10px; font: inherit; }
    button { min-height: 44px; border: 0; border-radius: 8px; background: #2563eb; color: white; font: inherit; font-weight: 750; cursor: pointer; }
    .result { min-height: 184px; display: grid; place-content: center; gap: 8px; background: linear-gradient(135deg,#eaf2ff,#fff7ed); border-radius: 8px; text-align: center; padding: 18px; }
    .value { font-size: 42px; font-weight: 820; }
    .muted { color: #64748b; }
    .hint { color: #64748b; font-size: 12px; line-height: 1.35; font-weight: 500; }
    .error { color: #b91c1c; font-weight: 700; }
    @media (max-width: 860px) { main, .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header><div class="wrap top"><h1>Оценка кредитного риска</h1><span class="muted">model.joblib</span></div></header>
  <main class="wrap">
    <section>
      <h2>Параметры заявки</h2>
      <form id="form">
        <div class="grid">
          <label>Доход<input id="AMT_INCOME_TOTAL" type="number" min="1" step="1" value="202500"></label>
          <label>Сумма кредита<input id="AMT_CREDIT" type="number" min="1" step="1" value="406597"></label>
          <label>Ежемесячный платёж<input id="AMT_ANNUITY" type="number" min="1" step="1" value="24700"></label>
          <label>Дата рождения<input id="BIRTH_DATE" type="date" value="1998-06-01"></label>
          <label>Дата начала работы<input id="EMPLOYMENT_START_DATE" type="date" value="2024-07-01"></label>
          <label>Количество детей<input id="CNT_CHILDREN" type="number" min="0" value="0"></label>
          <label>Пол
            <select id="CODE_GENDER">
              <option value="M">Мужской</option>
              <option value="F">Женский</option>
            </select>
          </label>
          <label>Есть машина
            <select id="FLAG_OWN_CAR">
              <option value="N">Нет</option>
              <option value="Y">Да</option>
            </select>
          </label>
          <label>Тип дохода
            <select id="NAME_INCOME_TYPE">
              <option value="Working">Работа по найму</option>
              <option value="Commercial associate">Коммерческий сотрудник</option>
              <option value="Pensioner">Пенсионер</option>
              <option value="State servant">Госслужащий</option>
            </select>
          </label>
          <label>Образование
            <select id="NAME_EDUCATION_TYPE">
              <option value="Secondary / secondary special">Среднее / среднее специальное</option>
              <option value="Higher education">Высшее</option>
              <option value="Incomplete higher">Неоконченное высшее</option>
            </select>
          </label>
          <label>Семейное положение
            <select id="NAME_FAMILY_STATUS">
              <option value="Single / not married">Не женат / не замужем</option>
              <option value="Married">В браке</option>
              <option value="Civil marriage">Гражданский брак</option>
              <option value="Separated">Раздельно</option>
            </select>
          </label>
          <label>Тип жилья
            <select id="NAME_HOUSING_TYPE">
              <option value="House / apartment">Дом / квартира</option>
              <option value="With parents">С родителями</option>
              <option value="Rented apartment">Аренда</option>
              <option value="Municipal apartment">Муниципальное жильё</option>
            </select>
          </label>
        </div>
        <button type="submit">Рассчитать риск</button>
      </form>
    </section>
    <section>
      <h2>Результат</h2>
      <div class="result">
        <div class="muted">Риск невозврата кредита</div>
        <div class="value" id="probability">-</div>
        <div id="category" class="muted">Заполните форму и запустите расчёт.</div>
        <div class="hint">Это оценка вероятности, что заявка попадёт в класс дефолта. Чем выше процент, тем выше риск для банка.</div>
        <div id="error" class="error"></div>
      </div>
    </section>
  </main>
  <script>
    const categorical = {
      NAME_CONTRACT_TYPE: "Cash loans",
      WEEKDAY_APPR_PROCESS_START: "MONDAY",
      ORGANIZATION_TYPE: "Business Entity Type 3"
    };
    const numeric = ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","CNT_CHILDREN"];
    const selects = ["CODE_GENDER","FLAG_OWN_CAR","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE"];

    function daysAgo(dateValue) {
      const selected = new Date(dateValue + "T00:00:00");
      const today = new Date();
      const current = new Date(today.getFullYear(), today.getMonth(), today.getDate());
      return -Math.max(0, Math.round((current - selected) / 86400000));
    }

    document.getElementById("form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const payload = {...categorical};
      numeric.forEach((name) => payload[name] = Number(document.getElementById(name).value));
      selects.forEach((name) => payload[name] = document.getElementById(name).value);
      payload.DAYS_BIRTH = daysAgo(document.getElementById("BIRTH_DATE").value);
      payload.DAYS_EMPLOYED = daysAgo(document.getElementById("EMPLOYMENT_START_DATE").value);
      payload.HOUR_APPR_PROCESS_START = 10;
      payload.EXT_SOURCE_2 = 0.26;
      payload.EXT_SOURCE_3 = 0.14;
      const response = await fetch("/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
      const data = await response.json();
      if (!response.ok) { document.getElementById("error").textContent = data.detail || "Ошибка"; return; }
      document.getElementById("probability").textContent = (data.default_probability * 100).toFixed(1) + "%";
      const labels = {low: "низкий", medium: "средний", high: "высокий"};
      document.getElementById("category").textContent = "Категория риска: " + (labels[data.risk_category] || data.risk_category);
      document.getElementById("error").textContent = "";
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UI_HTML


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        load_model()
        status = "ok"
    except FileNotFoundError:
        status = "model_missing"
    return {"status": status, "version": __version__}


@app.get("/metrics")
def metrics() -> dict[str, float | int]:
    requests_total = int(service_metrics["requests_total"])
    duration_sum = float(service_metrics["request_duration_seconds_sum"])
    average_duration = duration_sum / requests_total if requests_total else 0.0
    return {
        **service_metrics,
        "request_duration_seconds_avg": average_duration,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    metadata_path = resolve_path(artifact_cfg["metadata_path"])
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Метаданные модели не найдены. Запустите `python -m src.train`.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    try:
        return predict_one(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict-batch")
def batch_predict(payload: BatchPredictionRequest) -> dict[str, Any]:
    try:
        return {"predictions": predict_batch([record.model_dump() for record in payload.records])}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
