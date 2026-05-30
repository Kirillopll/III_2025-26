from __future__ import annotations

import json
from pathlib import Path


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def main() -> None:
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(exist_ok=True)
    eda = notebook(
        [
            md("# EDA по Home Credit Default Risk\n\nБазовый анализ данных для проекта кредитного скоринга."),
            code("import pandas as pd\n\ndata = pd.read_csv('../data/raw/application_train.csv', nrows=50000)\ndata.shape"),
            code("data['TARGET'].value_counts(normalize=True)"),
            code("data.isna().mean().sort_values(ascending=False).head(20)"),
        ]
    )
    experiments = notebook(
        [
            md("# Эксперименты с моделями\n\nНоутбук фиксирует основной запуск обучения и метрики."),
            code("!python -m src.train"),
            code("import pandas as pd\npd.read_csv('../artifacts/metrics.csv')"),
        ]
    )
    (notebooks_dir / "01_eda.ipynb").write_text(json.dumps(eda, ensure_ascii=False, indent=2), encoding="utf-8")
    (notebooks_dir / "02_model_experiments.ipynb").write_text(json.dumps(experiments, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
