from __future__ import annotations

from fastapi.testclient import TestClient

from churn.api import app


client = TestClient(app)

HIGH_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 180.0,
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_available"] == "true"


def test_model_info_exposes_metrics():
    response = client.get("/model-info")
    body = response.json()

    assert response.status_code == 200
    assert body["dataset_rows"] == 7043
    assert body["metrics"]["roc_auc"] >= 0.80


def test_high_risk_customer_returns_probability():
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    body = response.json()

    assert response.status_code == 200
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}


def test_predict_rejects_missing_required_field():
    payload = dict(HIGH_RISK_CUSTOMER)
    del payload["Contract"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
