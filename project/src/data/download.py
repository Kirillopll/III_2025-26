from __future__ import annotations


def download_dataset() -> None:
    raise FileNotFoundError(
        "Файл data/raw/application_train.csv не найден. "
        "Скачайте Home Credit Default Risk с Kaggle и распакуйте CSV-файлы в папку data/raw."
    )
