# Итоговый проект: сервис оценки кредитного риска

В этой папке находится сквозной мини-проект по курсу "Инженерия Искусственного Интеллекта".
Проект сделан как простой end-to-end ML-сервис: данные с Kaggle -> обучение нескольких моделей -> выбор лучшей -> REST API -> тесты и Docker.

---

## 1. Паспорт проекта

- **Название проекта:** Сервис оценки кредитного риска
- **Тема:** кредитный/риск-скоринг по табличным данным
- **Автор:** Кирилл
- **Датасет:** Home Credit Default Risk с Kaggle
- **Тип задачи:** бинарная классификация

**Краткое описание:**

Сервис оценивает вероятность дефолта клиента по признакам кредитной заявки.
На вход подаются поля из `application_train.csv`, на выходе возвращаются вероятность дефолта и категория риска: `low`, `medium` или `high`.
В проекте используются классические ML-модели из `scikit-learn`, FastAPI для сервиса и базовые Prometheus-метрики.

---

## 2. Структура проекта

```text
project/
  artifacts/                # обученная модель и JSON-отчёты, не коммитятся
  configs/                  # YAML-конфиг и пример переменных окружения
  data/
    raw/                    # сюда кладётся датасет Kaggle, не коммитится
    processed/              # подготовленные данные, если понадобятся
  notebooks/                # место для EDA-ноутбуков
  scripts/                  # вспомогательные скрипты
  src/credit_risk_service/  # основной код проекта
    api/                    # FastAPI-приложение
    core/                   # конфигурация и логирование
    ml/                     # обучение, признаки, модуль предсказаний
  tests/                    # pytest-тесты
  Dockerfile
  docker-compose.yml
  pyproject.toml
  requirements.txt
  report.md
  self-checklist.md
```

---

## 3. Требования и установка

Рекомендуется Python `3.10`-`3.12`. Если на компьютере установлен Python `3.14`, лучше создать окружение на 3.12, потому что часть ML-библиотек может не иметь готовых колёс под 3.14.

```bash
cd /d "P:\учеба\ИИИ\REP\III_2025-26\project"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 4. Данные

Используется открытый датасет Kaggle:

<https://www.kaggle.com/competitions/home-credit-default-risk/data>

Нужный файл для текущей версии:

```text
data/raw/application_train.csv
```

Полный архив Kaggle можно распаковать в `data/raw/`. Большие CSV-файлы не нужно коммитить в Git.

Проверить, что данные лежат правильно:

```bash
python scripts/inspect_data.py
```

Скрипт выведет короткий профиль данных и сохранит его в:

```text
artifacts/data_profile.json
```

---

## 5. Запуск обучения модели

Быстрый запуск на части реального датасета:

```bash
python -m credit_risk_service.ml.train
```

По умолчанию берётся `20000` строк, чтобы обучение проходило быстро на обычном ноутбуке.
Количество строк можно изменить:

```bash
python -m credit_risk_service.ml.train --max-rows 50000
```

Для проверки без Kaggle можно использовать demo-режим:

```bash
python -m credit_risk_service.ml.train --demo --max-rows 2000
```

После обучения появятся файлы:

```text
artifacts/credit_risk_model.joblib
artifacts/credit_risk_model.metrics.json
```

В эксперименте сравниваются:

- `DummyClassifier` как базовая модель;
- `LogisticRegression` как логистическая регрессия;
- `RandomForestClassifier` как случайный лес.

Лучшая модель выбирается по `Average Precision`, так как классы в задаче дефолта несбалансированы.

---

## 6. Запуск сервиса

После обучения модели:

```bash
uvicorn credit_risk_service.api.main:app --reload
```

Сервис будет доступен на:

```text
http://127.0.0.1:8000
```

Основные эндпоинты:

- `GET /health` - проверка состояния сервиса;
- `POST /predict` - оценка кредитной заявки;
- `GET /model-info` - информация о модели и метриках;
- `GET /metrics` - метрики для Prometheus.

Пример запроса:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d @sample-request.json
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 7. Docker

```bash
docker compose up --build
```

Перед запуском Docker нужно обучить модель, чтобы файл `artifacts/credit_risk_model.joblib` уже существовал.

---

## 8. Тесты

```bash
pytest
```

Проверяется:

- запуск API через FastAPI TestClient;
- эндпоинт `/health`;
- эндпоинт `/predict`;
- эндпоинт `/model-info`;
- логика категорий риска;
- минимальный demo-пайплайн обучения.

---

## 9. Демонстрация на защите

План демонстрации:

1. Показать структуру проекта и файл `README.md`.
2. Запустить `python scripts/inspect_data.py` и показать профиль датасета.
3. Запустить обучение `python -m credit_risk_service.ml.train`.
4. Показать файл метрик `artifacts/credit_risk_model.metrics.json`.
5. Запустить API через `uvicorn credit_risk_service.api.main:app --reload`.
6. Открыть Swagger UI и отправить запрос на `/predict`.
7. Показать `/health`, `/model-info` и `/metrics`.

---

## 10. Ограничения и развитие

Текущая версия специально сделана простой и воспроизводимой:

- используется только основная таблица `application_train.csv`;
- дополнительные таблицы Kaggle пока не агрегируются;
- гиперпараметры заданы вручную в `configs/model.yaml`;
- нет полноценного MLflow-трекинга.

Что можно улучшить дальше:

- добавить агрегированные признаки из `bureau.csv` и `previous_application.csv`;
- добавить CatBoost/LightGBM;
- сделать EDA-ноутбук с графиками;
- добавить SHAP или feature importance;
- логировать эксперименты в MLflow.
