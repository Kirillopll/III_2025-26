from __future__ import annotations

import pytest


@pytest.fixture
def sample_payload() -> dict:
    return {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 202500.0,
        "AMT_CREDIT": 406597.5,
        "AMT_ANNUITY": 24700.5,
        "DAYS_BIRTH": -9461,
        "DAYS_EMPLOYED": -637,
        "HOUR_APPR_PROCESS_START": 10,
        "EXT_SOURCE_2": 0.26,
        "EXT_SOURCE_3": 0.14,
    }
