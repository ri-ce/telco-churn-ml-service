from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from churn.data import FEATURES, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_has_expected_rows_and_columns():
    data = load_dataset(PROJECT_ROOT / "data" / "telco_churn.csv")

    assert len(data) == 7043
    assert set(FEATURES) <= set(data.columns)
    assert set(data["Churn"].unique()) == {0, 1}


def test_total_charges_is_numeric():
    data = load_dataset(PROJECT_ROOT / "data" / "telco_churn.csv")
    assert pd.api.types.is_numeric_dtype(data["TotalCharges"])


def test_trained_model_and_metrics_are_available():
    model = joblib.load(
        PROJECT_ROOT / "models" / "churn_pipeline.joblib"
    )
    metrics = json.loads(
        (PROJECT_ROOT / "models" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert hasattr(model, "predict_proba")
    assert metrics["roc_auc"] >= 0.80
    assert metrics["dataset_rows"] == 7043
