"""Dataset loading and feature definitions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Churn"


def load_dataset(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].astype(str).str.strip(),
        errors="coerce",
    )
    data["SeniorCitizen"] = data["SeniorCitizen"].astype(str)
    data[TARGET] = data[TARGET].map({"No": 0, "Yes": 1})
    if data[TARGET].isna().any():
        raise ValueError("Target column contains values outside Yes/No")
    return data


def prepare_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    return data[FEATURES].copy(), data[TARGET].astype(int)
