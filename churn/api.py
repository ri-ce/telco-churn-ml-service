"""FastAPI service for customer churn prediction."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from churn.schemas import (
    CustomerFeatures,
    ModelInfoResponse,
    PredictionResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv(
        "CHURN_MODEL_PATH",
        PROJECT_ROOT / "models" / "churn_pipeline.joblib",
    )
)
METRICS_PATH = Path(
    os.getenv(
        "CHURN_METRICS_PATH",
        PROJECT_ROOT / "models" / "metrics.json",
    )
)

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predict customer churn probability from structured features.",
)


def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model file is not available. Run the training script.",
        )
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict[str, object]:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_available": str(MODEL_PATH.exists()).lower(),
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    metrics = load_metrics()
    if not metrics:
        raise HTTPException(status_code=503, detail="Model metadata not found")
    return ModelInfoResponse(
        model_name=str(metrics["model_name"]),
        trained_at=str(metrics["trained_at"]),
        dataset_rows=int(metrics["dataset_rows"]),
        metrics={
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    model = load_model()
    frame = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(frame)[0, 1])
    if probability < 0.4:
        risk_level = "low"
    elif probability < 0.7:
        risk_level = "medium"
    else:
        risk_level = "high"
    metrics = load_metrics()
    return PredictionResponse(
        churn_probability=round(probability, 4),
        predicted_churn=probability >= 0.5,
        risk_level=risk_level,
        model_version=str(metrics.get("trained_at", "unknown")),
    )
