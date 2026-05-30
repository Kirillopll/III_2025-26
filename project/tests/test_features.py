from __future__ import annotations

import pandas as pd

from src.features.build_features import get_feature_spec, payload_to_frame, prepare_supervised_frame


def test_feature_spec_has_expected_columns() -> None:
    spec = get_feature_spec()
    assert "AMT_CREDIT" in spec.numeric
    assert "NAME_CONTRACT_TYPE" in spec.categorical
    assert spec.target == "TARGET"


def test_payload_to_frame(sample_payload: dict) -> None:
    frame = payload_to_frame(sample_payload)
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape[0] == 1
    assert "AMT_CREDIT" in frame.columns


def test_prepare_supervised_frame(sample_payload: dict) -> None:
    row = {**sample_payload, "TARGET": 0}
    X, y = prepare_supervised_frame(pd.DataFrame([row]))
    assert len(X) == 1
    assert int(y.iloc[0]) == 0
