# data

Папка с данными и демонстрационными запросами.

```text
data/
  raw/                         # исходные CSV-файлы Kaggle
  sample_application.csv        # небольшая выборка, создаётся после обучения
  sample_predict.json           # пример одного запроса к /predict
  sample_batch_predict.json     # пример запроса к /predict-batch
```

Для обучения нужен файл:

```text
data/raw/application_train.csv
```

Датасет скачивается с Kaggle Home Credit Default Risk:

```text
https://www.kaggle.com/competitions/home-credit-default-risk/data
```

Большие CSV-файлы из `data/raw/` не нужно добавлять в Git.
