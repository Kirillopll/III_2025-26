from __future__ import annotations

from argparse import Namespace

from credit_risk_service.ml.train import train


def test_demo_training_pipeline() -> None:
    config = {
        "data": {
            "target_column": "TARGET",
            "id_columns": ["SK_ID_CURR"],
            "test_size": 0.25,
            "random_state": 42,
        },
        "training": {
            "selection_metric": "average_precision",
            "models": {
                "dummy": {"enabled": True},
                "logistic_regression": {"enabled": True, "max_iter": 200, "class_weight": "balanced"},
                "random_forest": {"enabled": False},
            },
        },
        "service": {"model_path": "models/test.joblib"},
    }
    args = Namespace(demo=True, data=None, output=None, max_rows=300)

    artifact = train(config, args)

    assert artifact["model_name"] in {"dummy", "logistic_regression"}
    assert artifact["target_column"] == "TARGET"
    assert artifact["feature_columns"]
    assert "average_precision" in artifact["metrics"]
