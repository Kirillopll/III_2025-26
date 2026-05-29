# src

Основной код проекта находится в пакете `credit_risk_service`.

```text
credit_risk_service/
  api/   # FastAPI-приложение и pydantic-схемы
  core/  # конфигурация и логирование
  ml/    # обучение, пайплайн признаков и модуль предсказаний
```

Главные команды:

```bash
python -m credit_risk_service.ml.train
uvicorn credit_risk_service.api.main:app --reload
```
