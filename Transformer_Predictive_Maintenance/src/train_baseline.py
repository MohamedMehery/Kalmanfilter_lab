"""
Classical ML baseline for transformer predictive maintenance.

Trains and compares Logistic Regression, Random Forest and Gradient
Boosting classifiers on the engineered feature set, selects the best
model on the dev set, reports final metrics on the held-out test set,
and saves the winning model + scaler for later use / MCU export.

Usage:
    python src/train_baseline.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, TARGET_COL


def load_split(name: str):
    path = DATA_PROCESSED_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    feature_cols = (DATA_PROCESSED_DIR / "feature_columns.txt").read_text().splitlines()
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    return X, y, feature_cols


def evaluate(model, X, y, scaler=None) -> dict:
    Xs = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(Xs)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "report": classification_report(y, pred, zero_division=0, output_dict=True),
    }


def main() -> None:
    X_train, y_train, feature_cols = load_split("train")
    X_dev, y_dev, _ = load_split("dev")
    X_test, y_test, _ = load_split("test")

    print(f"Train: {X_train.shape}, positives={y_train.mean():.2%}")
    print(f"Dev:   {X_dev.shape}, positives={y_dev.mean():.2%}")
    print(f"Test:  {X_test.shape}, positives={y_test.mean():.2%}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    dev_results = {}
    fitted_models = {}
    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        fitted_models[name] = model
        metrics = evaluate(model, X_dev, y_dev, scaler)
        dev_results[name] = metrics
        print(
            f"[dev] {name:>20s} | acc={metrics['accuracy']:.3f} "
            f"f1={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f} "
            f"pr_auc={metrics['pr_auc']:.3f}"
        )

    best_name = max(dev_results, key=lambda n: dev_results[n]["f1"])
    best_model = fitted_models[best_name]
    print(f"\nBest model on dev set (by F1): {best_name}")

    test_metrics = evaluate(best_model, X_test, y_test, scaler)
    print(
        f"[test] {best_name:>20s} | acc={test_metrics['accuracy']:.3f} "
        f"f1={test_metrics['f1']:.3f} roc_auc={test_metrics['roc_auc']:.3f} "
        f"pr_auc={test_metrics['pr_auc']:.3f}"
    )
    print(json.dumps(test_metrics["report"], indent=2))

    # --------------------------------------------------------------------
    # Persist artifacts
    # --------------------------------------------------------------------
    joblib.dump(best_model, MODELS_DIR / "baseline_best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "baseline_scaler.joblib")
    with open(MODELS_DIR / "baseline_feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    summary = {
        "best_model": best_name,
        "dev_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "report"} for k, v in dev_results.items()},
        "test_metrics": {kk: vv for kk, vv in test_metrics.items() if kk != "report"},
        "test_classification_report": test_metrics["report"],
    }
    with open(REPORTS_DIR / "baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved best model -> {MODELS_DIR / 'baseline_best_model.joblib'}")
    print(f"Saved scaler -> {MODELS_DIR / 'baseline_scaler.joblib'}")
    print(f"Saved summary -> {REPORTS_DIR / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
