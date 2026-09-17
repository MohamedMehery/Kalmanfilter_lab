"""
Deep-learning model for transformer predictive maintenance.

A deliberately small, MCU-friendly Multi-Layer Perceptron (MLP) trained
with TensorFlow/Keras. Architecture is intentionally tiny (dense
28 -> 16 -> 8 -> 1) so it can be quantized to int8 and run on an
Arduino / MCU with a few KB of RAM.

Usage:
    python src/train_dl_model.py

Outputs:
    models/dl_model.keras            (Keras model, for re-loading in Python)
    models/dl_model.tflite            (Float32 TFLite model)
    models/dl_model_quant.tflite      (int8-quantized TFLite model)
    models/dl_scaler.joblib           (StandardScaler fit on train split)
    reports/dl_results.json           (train/dev/test metrics)
    reports/figures/dl_training_curves.png
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COL,
)

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_split(name: str):
    path = DATA_PROCESSED_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    feature_cols = (DATA_PROCESSED_DIR / "feature_columns.txt").read_text().splitlines()
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return X, y, feature_cols


def build_model(n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="transformer_fault_mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def evaluate(model, X, y) -> dict:
    proba = model.predict(X, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "report": classification_report(y, pred, zero_division=0, output_dict=True),
    }


def representative_dataset_gen(X_train_s: np.ndarray):
    """Yields samples for TFLite int8 post-training quantization calibration."""
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(X_train_s), size=min(200, len(X_train_s)), replace=False)
    for i in idx:
        yield [X_train_s[i : i + 1].astype(np.float32)]


def convert_to_tflite(model: tf.keras.Model, X_train_s: np.ndarray) -> tuple[bytes, bytes]:
    # Float32 TFLite (portable, still much smaller than the Keras model).
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float = converter.convert()

    # int8 quantized TFLite (best for MCU deployment: ~4x smaller, integer-only math).
    converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_q.representative_dataset = lambda: representative_dataset_gen(X_train_s)
    converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_q.inference_input_type = tf.int8
    converter_q.inference_output_type = tf.int8
    tflite_quant = converter_q.convert()

    return tflite_float, tflite_quant


def plot_training_curves(history: tf.keras.callbacks.History) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="dev")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["auc"], label="train")
    axes[1].plot(history.history["val_auc"], label="dev")
    axes[1].set_title("ROC-AUC")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.suptitle("Transformer Fault MLP — training curves")
    fig.tight_layout()
    out_path = FIGURES_DIR / "dl_training_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved training curves -> {out_path}")


def main() -> None:
    X_train, y_train, feature_cols = load_split("train")
    X_dev, y_dev, _ = load_split("dev")
    X_test, y_test, _ = load_split("test")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_dev_s = scaler.transform(X_dev).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # Class weighting to counter imbalance (positives ~10-13%).
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: 1.0, 1: float(n_neg / max(n_pos, 1))}
    print(f"Class weights: {class_weight}")

    model = build_model(n_features=X_train_s.shape[1])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
        ),
    ]

    history = model.fit(
        X_train_s,
        y_train,
        validation_data=(X_dev_s, y_dev),
        epochs=150,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_curves(history)

    dev_metrics = evaluate(model, X_dev_s, y_dev)
    test_metrics = evaluate(model, X_test_s, y_test)
    print(
        f"[dev]  acc={dev_metrics['accuracy']:.3f} f1={dev_metrics['f1']:.3f} "
        f"roc_auc={dev_metrics['roc_auc']:.3f} pr_auc={dev_metrics['pr_auc']:.3f}"
    )
    print(
        f"[test] acc={test_metrics['accuracy']:.3f} f1={test_metrics['f1']:.3f} "
        f"roc_auc={test_metrics['roc_auc']:.3f} pr_auc={test_metrics['pr_auc']:.3f}"
    )
    print(json.dumps(test_metrics["report"], indent=2))

    # --------------------------------------------------------------------
    # Persist artifacts
    # --------------------------------------------------------------------
    model.save(MODELS_DIR / "dl_model.keras")
    joblib.dump(scaler, MODELS_DIR / "dl_scaler.joblib")
    with open(MODELS_DIR / "dl_feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    tflite_float, tflite_quant = convert_to_tflite(model, X_train_s)
    (MODELS_DIR / "dl_model.tflite").write_bytes(tflite_float)
    (MODELS_DIR / "dl_model_quant.tflite").write_bytes(tflite_quant)
    print(f"Saved float32 TFLite model ({len(tflite_float)} bytes)")
    print(f"Saved int8 quantized TFLite model ({len(tflite_quant)} bytes)")

    # Save scaler params (mean/scale) in plain JSON too, so the MCU folder
    # doesn't need joblib/pickle to reconstruct pre-processing in C/C++.
    scaler_json = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_columns": feature_cols,
    }
    with open(MODELS_DIR / "dl_scaler.json", "w") as f:
        json.dump(scaler_json, f, indent=2)

    summary = {
        "architecture": "Dense(28->16->8->1), sigmoid output",
        "dev_metrics": {k: v for k, v in dev_metrics.items() if k != "report"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "report"},
        "test_classification_report": test_metrics["report"],
        "tflite_float_bytes": len(tflite_float),
        "tflite_quant_bytes": len(tflite_quant),
    }
    with open(REPORTS_DIR / "dl_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved Keras model -> {MODELS_DIR / 'dl_model.keras'}")
    print(f"Saved TFLite models -> {MODELS_DIR}")
    print(f"Saved summary -> {REPORTS_DIR / 'dl_results.json'}")


if __name__ == "__main__":
    main()
