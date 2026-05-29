from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Собрать краткий профиль качества данных для application_train.csv.")
    parser.add_argument("--data", default="data/raw/application_train.csv", help="Путь к application_train.csv.")
    parser.add_argument("--output", default="artifacts/data_profile.json", help="Путь для сохранения JSON-профиля.")
    parser.add_argument("--sample-rows", type=int, default=50000, help="Число строк для оценки пропусков.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Датасет не найден: {data_path}")

    full_header = pd.read_csv(data_path, nrows=0)
    sample = pd.read_csv(data_path, nrows=args.sample_rows)
    target = sample["TARGET"]

    missing_share = sample.isna().mean().sort_values(ascending=False).head(15)
    profile = {
        "dataset_path": str(data_path),
        "sample_rows": int(len(sample)),
        "total_columns": int(len(full_header.columns)),
        "target_positive_share_sample": float(target.mean()),
        "numeric_columns_sample": int(len(sample.select_dtypes(include=["number", "bool"]).columns)),
        "categorical_columns_sample": int(len(sample.columns) - len(sample.select_dtypes(include=["number", "bool"]).columns)),
        "top_missing_share": {column: float(value) for column, value in missing_share.items()},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print(f"Профиль данных сохранён: {output_path.resolve()}")


if __name__ == "__main__":
    main()
