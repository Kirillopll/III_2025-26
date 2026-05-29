from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from credit_risk_service.core.config import DEFAULT_CONFIG_PATH, load_yaml_config, resolve_project_path
from credit_risk_service.ml.features import build_preprocessor, split_feature_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение моделей кредитного риск-скоринга.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Путь к YAML-конфигу.")
    parser.add_argument("--data", default=None, help="Переопределить путь к application_train.csv.")
    parser.add_argument("--output", default=None, help="Переопределить путь сохранения артефакта модели.")
    parser.add_argument("--max-rows", type=int, default=None, help="Ограничить число строк для быстрого локального запуска.")
    parser.add_argument("--demo", action="store_true", help="Обучить модель на сгенерированном demo-наборе.")
    return parser.parse_args()


def make_demo_dataset(rows: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = pd.Series(range(rows)).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    data = pd.DataFrame(
        {
            "SK_ID_CURR": 100000 + rng,
            "AMT_INCOME_TOTAL": 60000 + (rng % 120) * 2500,
            "AMT_CREDIT": 100000 + (rng % 200) * 4500,
            "AMT_ANNUITY": 7000 + (rng % 80) * 300,
            "DAYS_BIRTH": -22000 + (rng % 10000),
            "DAYS_EMPLOYED": -5000 + (rng % 4500),
            "CNT_CHILDREN": rng % 4,
            "CODE_GENDER": rng.map(lambda value: "M" if value % 3 == 0 else "F"),
            "FLAG_OWN_CAR": rng.map(lambda value: "Y" if value % 4 == 0 else "N"),
            "NAME_CONTRACT_TYPE": rng.map(lambda value: "Revolving loans" if value % 7 == 0 else "Cash loans"),
        }
    )
    debt_ratio = data["AMT_ANNUITY"] / data["AMT_INCOME_TOTAL"]
    young = data["DAYS_BIRTH"] > -12000
    unstable_job = data["DAYS_EMPLOYED"] > -800
    data["TARGET"] = ((debt_ratio > 0.22) | (young & unstable_job) | (data["NAME_CONTRACT_TYPE"] == "Revolving loans")).astype(int)
    return data


def load_training_data(config: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    if args.demo:
        max_rows = args.max_rows or 2000
        return make_demo_dataset(rows=max_rows, random_state=config["data"].get("random_state", 42))

    data_path = resolve_project_path(args.data or config["data"]["raw_path"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл с обучающими данными не найден: {data_path}. Скачайте Home Credit Default Risk и положите "
            "`application_train.csv` в `project/data/raw/` или запустите обучение с флагом `--demo`."
        )

    max_rows = args.max_rows
    if max_rows is None:
        max_rows = config["data"].get("max_rows")
    return pd.read_csv(data_path, nrows=max_rows)


def build_candidate_models(config: dict[str, Any], numeric_features: list[str], categorical_features: list[str]) -> dict[str, Pipeline]:
    models_config = config.get("training", {}).get("models", {})
    candidates: dict[str, Pipeline] = {}

    def pipeline(model: Any) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", model),
            ]
        )

    if models_config.get("dummy", {}).get("enabled", True):
        candidates["dummy"] = pipeline(DummyClassifier(strategy="prior"))

    logistic_config = models_config.get("logistic_regression", {})
    if logistic_config.get("enabled", True):
        candidates["logistic_regression"] = pipeline(
            LogisticRegression(
                max_iter=int(logistic_config.get("max_iter", 1000)),
                class_weight=logistic_config.get("class_weight", "balanced"),
                n_jobs=None,
            )
        )

    forest_config = models_config.get("random_forest", {})
    if forest_config.get("enabled", True):
        candidates["random_forest"] = pipeline(
            RandomForestClassifier(
                n_estimators=int(forest_config.get("n_estimators", 200)),
                max_depth=forest_config.get("max_depth", 12),
                class_weight=forest_config.get("class_weight", "balanced_subsample"),
                random_state=config["data"].get("random_state", 42),
                n_jobs=-1,
            )
        )

    return candidates


def evaluate_model(model: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "average_precision": float(average_precision_score(y_valid, probabilities)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
    }


def train(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data = load_training_data(config, args)
    target_column = config["data"]["target_column"]
    id_columns = config["data"].get("id_columns", [])

    if target_column not in data.columns:
        raise ValueError(f"Целевая колонка `{target_column}` отсутствует в обучающих данных.")

    feature_columns = [column for column in data.columns if column not in {target_column, *id_columns}]
    x = data[feature_columns]
    y = data[target_column].astype(int)

    numeric_features, categorical_features = split_feature_types(data[feature_columns], ignored_columns=[])
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=config["data"].get("test_size", 0.2),
        stratify=y,
        random_state=config["data"].get("random_state", 42),
    )

    candidates = build_candidate_models(config, numeric_features, categorical_features)
    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in candidates.items():
        print(f"Обучение модели {model_name}...")
        model.fit(x_train, y_train)
        results[model_name] = evaluate_model(model, x_valid, y_valid)
        fitted_models[model_name] = model
        print(json.dumps({model_name: results[model_name]}, indent=2))

    selection_metric = config.get("training", {}).get("selection_metric", "average_precision")
    best_model_name = max(results, key=lambda name: results[name][selection_metric])
    best_model = fitted_models[best_model_name]

    return {
        "model": best_model,
        "model_name": best_model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": results[best_model_name],
        "all_metrics": results,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "selection_metric": selection_metric,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    artifact = train(config, args)

    output_path = resolve_project_path(args.output or config["service"]["model_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(artifact["all_metrics"], indent=2), encoding="utf-8")

    print(f"Лучшая модель: {artifact['model_name']}")
    print(f"Артефакт модели сохранён: {output_path}")
    print(f"Метрики сохранены: {metrics_path}")


if __name__ == "__main__":
    main()
