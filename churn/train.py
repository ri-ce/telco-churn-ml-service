"""Train and evaluate customer churn models."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_dataset,
    prepare_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(
    data_path: Path,
    model_path: Path,
    metrics_path: Path,
    chart_path: Path,
) -> dict[str, object]:
    data = load_dataset(data_path)
    features, target = prepare_features(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in candidate_models().items():
        scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        fitted_models[name] = model
        comparison[name] = {
            "cv_roc_auc_mean": round(float(np.mean(scores)), 4),
            "cv_roc_auc_std": round(float(np.std(scores)), 4),
        }

    best_name = max(
        comparison,
        key=lambda name: comparison[name]["cv_roc_auc_mean"],
    )
    best_model = fitted_models[best_name]
    probabilities = best_model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics: dict[str, object] = {
        "model_name": best_name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "dataset_rows": int(len(data)),
        "churn_rate": round(float(target.mean()), 4),
        "model_comparison": comparison,
        "dataset_sha256": dataset_hash(data_path),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(7.2, 5.2))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC={metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Customer churn ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=180)
    plt.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "telco_churn.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "churn_pipeline.joblib",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "models" / "metrics.json",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=PROJECT_ROOT / "docs" / "model-evaluation.png",
    )
    args = parser.parse_args()
    metrics = train(args.data, args.model, args.metrics, args.chart)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
