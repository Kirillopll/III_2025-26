# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

## 1. Паспорт проекта

- **Название проекта:** `Сервис оценки кредитного риска`
- **Автор:** 
- ФИО: Оплеухин Кирилл Романович
- Группа: БББО-23-24
- Контакт: @cyrusso
- **Задача:** предсказать вероятность дефолта клиента (`TARGET`) по признакам кредитной заявки.
- **Датасет:** Kaggle Home Credit Default Risk.
- **Результат:** REST API с `/predict`, `/predict-batch`, `/health`, `/metrics`, `/model-info` и простой HTML-страницей на `/`.

---

## 2. Структура проекта

```text
project/
  README.md
  report.md
  self-checklist.md
  SECURITY.md
  requirements.txt
  Dockerfile
  configs/
  data/
  notebooks/
  src/
  tests/
  artifacts/
  scripts/
```

Ключевые части:

- `src/data/` - проверка наличия и загрузка Home Credit CSV.
- `src/features/` - список признаков и подготовка обучающей таблицы.
- `src/models/` - обучение, сравнение моделей и инференс.
- `src/service/` - FastAPI-сервис с пользовательской страницей.
- `notebooks/` - EDA и эксперименты.
- `artifacts/` - модель, метрики, графики и тестовые предсказания.

---

## 3. Установка

Команды выполняются из корня папки `project`.

PowerShell без активации окружения, если `.venv` уже создан:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Если `.venv` отсутствует:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 4. Данные

В `data/raw/` должен лежать файл:

```text
application_train.csv
```

Можно положить туда весь архив Kaggle Home Credit Default Risk. В Git эти CSV не добавляются.

---

## 5. Обучение

```powershell
.\.venv\Scripts\python.exe -m src.train
```

Команда создаёт:

- `artifacts/model.joblib` - финальная модель;
- `artifacts/model_metadata.json` - описание модели и метрик;
- `artifacts/metrics.csv` и `artifacts/metrics.json` - сравнение моделей;
- `artifacts/test_predictions.csv` - предсказания на тестовой части;
- `artifacts/feature_importance.csv` - важность признаков;
- `artifacts/error_by_hour.csv` - средняя ошибка по часу подачи заявки;
- `artifacts/*.png` - графики для отчёта и защиты.

---

## 6. Запуск сервиса

```powershell
.\.venv\Scripts\python.exe -m src.service
```

Открыть в браузере:

```text
http://127.0.0.1:8000
```

Полезные адреса:

- `GET /` - простая пользовательская форма;
- `GET /health` - проверка состояния;
- `GET /metrics` - базовые метрики работы сервиса;
- `GET /model-info` - информация о модели;
- `POST /predict` - одно предсказание;
- `POST /predict-batch` - пакет предсказаний;
- `GET /docs` - Swagger UI.

В пользовательской форме показаны только понятные поля заявки. Технические признаки датасета Home Credit, например `EXT_SOURCE_2`, `EXT_SOURCE_3` и час подачи заявки, не выводятся на экран и подставляются сервисом как значения по умолчанию для демонстрационного расчёта.

Запрос из PowerShell:

```powershell
curl.exe -s -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@data/sample_predict.json"
```

Если порт `8000` занят:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.service.app:app --port 8001
```

---

## 7. Docker

```powershell
docker build -t credit-risk-project .
docker run -p 8000:8000 credit-risk-project
```

Перед сборкой выполните обучение, чтобы появился `artifacts/model.joblib`.
В Docker сервис запускается на `0.0.0.0:8000`, поэтому он доступен с компьютера по адресу `http://127.0.0.1:8000`.

---

## 8. Тесты

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

Тесты проверяют подготовку признаков, инференс и основные endpoints сервиса.

---

## 9. Демонстрация

1. Показать структуру проекта.
2. Запустить `python -m src.train`.
3. Показать `artifacts/metrics.csv` и графики.
4. Запустить `python -m src.service`.
5. Открыть `http://127.0.0.1:8000`.
6. Выполнить запрос на `/predict`.
7. Показать `/health` и `/metrics` как базовую наблюдаемость.
