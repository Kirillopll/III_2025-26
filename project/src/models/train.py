from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config, resolve_path
from src.data.load import load_application_data, write_sample
from src.features.build_features import get_feature_spec, prepare_supervised_frame


matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def split_dataset(X: pd.DataFrame, y: pd.Series, config: dict):
    split_cfg = config["split"]
    random_state = int(config["project"]["random_state"])
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=float(split_cfg["test_size"]),
        stratify=y,
        random_state=random_state,
    )
    relative_validation_size = float(split_cfg["validation_size"]) / (
        float(split_cfg["train_size"]) + float(split_cfg["validation_size"])
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_validation_size,
        stratify=y_train_val,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor() -> ColumnTransformer:
    spec = get_feature_spec()
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("one_hot", make_one_hot_encoder())])
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, spec.categorical),
            ("numeric", numeric_pipeline, spec.numeric),
        ]
    )


def build_models(config: dict) -> dict[str, Pipeline]:
    logistic_cfg = config["models"]["logistic_regression"]
    forest_cfg = config["models"]["random_forest"]
    return {
        "dummy_prior": Pipeline([("preprocess", build_preprocessor()), ("model", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=int(logistic_cfg["max_iter"]),
                        class_weight=str(logistic_cfg["class_weight"]),
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(forest_cfg["n_estimators"]),
                        max_depth=int(forest_cfg["max_depth"]),
                        min_samples_leaf=int(forest_cfg["min_samples_leaf"]),
                        n_jobs=int(forest_cfg["n_jobs"]),
                        class_weight="balanced_subsample",
                        random_state=int(config["project"]["random_state"]),
                    ),
                ),
            ]
        ),
    }


def classification_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
    }


def get_feature_names(model: Pipeline) -> list[str]:
    return list(model.named_steps["preprocess"].get_feature_names_out())


def save_feature_importance(model: Pipeline, path: Path) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    feature_names = get_feature_names(model)
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(feature_names))
    importance = pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False)
    importance.to_csv(path, index=False)
    return importance


def make_plots(data: pd.DataFrame, predictions: pd.DataFrame, feature_importance: pd.DataFrame, error_by_hour: pd.DataFrame, artifact_cfg: dict) -> None:
    sns.set_theme(style="whitegrid")

    hourly = data.groupby("HOUR_APPR_PROCESS_START", as_index=False)["TARGET"].mean()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=hourly, x="HOUR_APPR_PROCESS_START", y="TARGET", marker="o")
    plt.title("Average default rate by application hour")
    plt.xlabel("Hour")
    plt.ylabel("Default rate")
    plt.tight_layout()
    plt.savefig(resolve_path(artifact_cfg["default_by_hour_plot"]), dpi=150)
    plt.close()

    sample = predictions.head(600)
    jitter = np.random.default_rng(42).normal(0, 0.015, len(sample))
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=sample["actual"] + jitter, y=sample["default_probability"], alpha=0.55)
    plt.title("Actual target vs predicted default probability")
    plt.xlabel("Actual target")
    plt.ylabel("Predicted probability")
    plt.tight_layout()
    plt.savefig(resolve_path(artifact_cfg["prediction_plot"]), dpi=150)
    plt.close()

    top = feature_importance.head(15).copy()
    plt.figure(figsize=(9, 6))
    sns.barplot(data=top, x="importance", y="feature", color="#2f7ebc")
    plt.title("Top feature importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(resolve_path(artifact_cfg["feature_importance_plot"]), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=error_by_hour, x="HOUR_APPR_PROCESS_START", y="mae", color="#cc6b49")
    plt.title("Mean absolute probability error by application hour")
    plt.xlabel("Hour")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(resolve_path(artifact_cfg["error_by_hour_plot"]), dpi=150)
    plt.close()


def train_and_evaluate() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    artifact_cfg = config["artifacts"]
    artifact_dir = resolve_path(artifact_cfg["dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    data = load_application_data()
    write_sample(data)
    X, y = prepare_supervised_frame(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, config)

    models = build_models(config)
    results: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, model in models.items():
        logger.info("Обучение %s", name)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        val_prob = model.predict_proba(X_val)[:, 1]
        results.append({"model": name, "split": "validation", **classification_metrics(y_val, val_prob)})

    best_name = max(results, key=lambda row: row["average_precision"])["model"]
    best_model = fitted_models[best_name]
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_prob)
    results.append({"model": best_name, "split": "test", **test_metrics})

    joblib.dump(best_model, resolve_path(artifact_cfg["model_path"]))
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(resolve_path(artifact_cfg["metrics_csv"]), index=False)

    predictions = X_test.copy()
    predictions["actual"] = y_test.to_numpy()
    predictions["default_probability"] = test_prob
    predictions["absolute_error"] = (predictions["actual"] - predictions["default_probability"]).abs()
    predictions.to_csv(resolve_path(artifact_cfg["predictions_csv"]), index=False)

    error_by_hour = predictions.groupby("HOUR_APPR_PROCESS_START", as_index=False)["absolute_error"].mean().rename(columns={"absolute_error": "mae"})
    error_by_hour.to_csv(resolve_path(artifact_cfg["error_by_hour_csv"]), index=False)

    importance = save_feature_importance(best_model, resolve_path(artifact_cfg["feature_importance_csv"]))
    make_plots(data, predictions, importance, error_by_hour, artifact_cfg)

    metadata = {
        "project": config["project"]["name"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Kaggle Home Credit Default Risk, application_train.csv",
        "target": config["features"]["target"],
        "features": config["features"],
        "final_model": best_name,
        "validation_results": [row for row in results if row["split"] == "validation"],
        "test_metrics": test_metrics,
        "model_path": artifact_cfg["model_path"],
    }
    with resolve_path(artifact_cfg["metadata_path"]).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    with resolve_path(artifact_cfg["metrics_json"]).open("w", encoding="utf-8") as file:
        json.dump({"results": results, "metadata": metadata}, file, ensure_ascii=False, indent=2)
    logger.info("Финальная модель: %s, test metrics: %s", best_name, test_metrics)
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
