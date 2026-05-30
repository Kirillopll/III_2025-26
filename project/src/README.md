# src

Основной код проекта организован как в эталонном проекте:

```text
src/
  data/       # проверка наличия и загрузка данных
  features/   # подготовка признаков
  models/     # обучение модели и инференс
  service/    # FastAPI-сервис и HTML-страница
  config.py   # чтение configs/config.yaml
  train.py    # точка входа для обучения
```

Основные команды:

```bash
python -m src.train
python -m src.service
```
