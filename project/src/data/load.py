from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.data.download import download_dataset


def ensure_dataset() -> Path:
    config = load_config()
    train_csv = resolve_path(config["data"]["train_csv"])
    if not train_csv.exists():
        download_dataset()
    return train_csv


def load_application_data(max_rows: int | None = None) -> pd.DataFrame:
    train_csv = ensure_dataset()
    config = load_config()
    rows = max_rows if max_rows is not None else config["data"].get("max_rows")
    return pd.read_csv(train_csv, nrows=rows)


def write_sample(data: pd.DataFrame | None = None) -> Path:
    config = load_config()
    sample_path = resolve_path(config["data"]["sample_csv"])
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    source = load_application_data() if data is None else data
    sample = source.head(int(config["data"]["sample_size"]))
    sample.to_csv(sample_path, index=False)
    return sample_path
