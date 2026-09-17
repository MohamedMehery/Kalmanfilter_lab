"""
Multi-head TinyML model: transformer health classification + WTI forecasting.
=============================================================================

Architecture (chosen to stay inside TFLite-Micro's supported op set)

        Input (WINDOW=12, F=19)                 3 h of 15-min samples
              |
        Conv1D(16, k=3, ReLU)                   local thermal/electrical trend
              |
        Conv1D(16, k=3, ReLU)
              |
        MaxPool1D(2)
              |
        Flatten -> Dense(24, ReLU) -> Dropout
              |
        +-----+-----------------------+
        |                             |
    Dense(3) softmax              Dense(1) linear
    health state @ (t, t+H]       WTI @ t+H (standardised)

No LSTM/GRU: those lower to TFLM ops that either are unsupported or blow up the
tensor arena.  A stacked 1-D CNN captures the same short-horizon dynamics far
more cheaply.

Two things this script is deliberately careful about
----------------------------------------------------
1. MODEL SELECTION.  The rare CRITICAL class makes a class-weighted validation
   loss wildly unstable (a handful of confident mistakes swamp it), so the
   weighted loss is used for *gradients only*.  Checkpointing / early stopping
   are driven by dev-set macro-F1, which is what we actually care about.

2. HONEST BASELINE.  A transformer's health state is highly persistent: simply
   predicting "whatever the state is now" already scores ~96 % here.  Raw
   accuracy is therefore a misleading headline, so the script always prints
   the persistence baseline and the early-warning recall next to it.

Run:
    python 02_training/train.py
    python 02_training/train.py --epochs 250 --seed 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    ARTIFACT_DIR, CLASS_NAMES, MODEL, N_CLASSES, SPLIT_DIR,
)

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from keras import layers  # noqa: E402


# --------------------------------------------------------------- data io
def load_splits():
    f = np.load(SPLIT_DIR / "dataset.npz")
    meta = json.loads((SPLIT_DIR / "meta.json").read_text())
    return {k: f[k] for k in f.files}, meta


def fit_normaliser(X: np.ndarray):
    """Per-feature mean/std over the TRAIN split only (no leakage)."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


# --------------------------------------------------------------- metrics
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int = N_CLASSES) -> float:
    f1s = []
    for c in range(k):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def alarm_f1(y_true_cls: np.ndarray, risk: np.ndarray, thr: float = 0.5) -> float:
    """F1 of the binary ALARM event (state deteriorates within the horizon)."""
    a = (y_true_cls > 0).astype(int)
    p = (risk >= thr).astype(int)
    tp = int(((p == 1) & (a == 1)).sum())
    fp = int(((p == 1) & (a == 0)).sum())
    fn = int(((p == 0) & (a == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def best_alarm_f1(y_true_cls: np.ndarray, risk: np.ndarray) -> tuple[float, float]:
    best_f1, best_thr = -1.0, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        f = alarm_f1(y_true_cls, risk, float(thr))
        if f > best_f1:
            best_f1, best_thr = f, float(thr)
    return best_f1, best_thr


class DevSelector(keras.callbacks.Callback):
    """
    Model selection on the DEV split.

    Selects on the *product* metric -- the threshold-optimised F1 of the alarm
    event -- rather than on val_loss (which the class weights make unstable)
    or on 3-class macro-F1 (which is dominated by persistence and barely moves
    between good and bad models).
    """

    def __init__(self, X, y, patience: int):
        super().__init__()
        self.X, self.y, self.patience = X, y, patience
        self.best, self.best_w, self.best_epoch, self.wait = -1.0, None, 0, 0
        self.best_thr = 0.5
        self.history: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        prob = self.model.predict(self.X, verbose=0)[0]
        f1, thr = best_alarm_f1(self.y, prob[:, 1] + prob[:, 2])
        self.history.append(f1)
        (logs := logs if logs is not None else {})["val_alarm_f1"] = f1
        if f1 > self.best + 1e-4:
            self.best, self.best_epoch, self.wait = f1, epoch, 0
            self.best_thr = thr
            self.best_w = [w.copy() for w in self.model.get_weights()]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\n[train] early stop: no dev alarm-F1 gain for "
                      f"{self.patience} epochs")

    def on_train_end(self, logs=None):
        if self.best_w is not None:
            self.model.set_weights(self.best_w)
            print(f"[train] restored epoch {self.best_epoch+1} "
                  f"(dev alarm-F1 = {self.best:.4f} @ thr {self.best_thr:.2f})")


# ----------------------------------------------------------------- model
def build_model(window: int, n_feat: int, cfg=MODEL) -> keras.Model:
    reg = keras.regularizers.l2(cfg["l2"])
    inp = keras.Input(shape=(window, n_feat), name="window")

    x = layers.Conv1D(cfg["conv_filters"], cfg["conv_kernel"], padding="valid",
                      activation="relu", kernel_regularizer=reg, name="conv1")(inp)
    x = layers.Conv1D(cfg["conv_filters"], cfg["conv_kernel"], padding="valid",
                      activation="relu", kernel_regularizer=reg, name="conv2")(x)
    x = layers.MaxPooling1D(2, name="pool")(x)
    x = layers.Flatten(name="flat")(x)
    x = layers.Dense(cfg["dense_units"], activation="relu",
                     kernel_regularizer=reg, name="trunk")(x)
    x = layers.Dropout(cfg["dropout"], name="drop")(x)

    out_cls = layers.Dense(N_CLASSES, activation="softmax", name="health")(x)
    out_reg = layers.Dense(1, activation=None, name="wti")(x)
    return keras.Model(inp, [out_cls, out_reg], name="transformer_pdm")


def class_weights(y: np.ndarray, power: float = 0.5) -> dict[int, float]:
    """
    Square-root balanced weights.

    Fully balanced (power=1.0) weights hit ~50x here and make the model
    hallucinate CRITICAL everywhere, destroying precision.  The sqrt variant
    lifts recall on the rare classes without wrecking the majority class.
    """
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    w = (counts.sum() / (len(counts) * counts)) ** power
    w = np.clip(w, 0.5, 10.0)
    return {i: float(v) for i, v in enumerate(w)}


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=MODEL["epochs"])
    ap.add_argument("--batch-size", type=int, default=MODEL["batch_size"])
    ap.add_argument("--seed", type=int, default=MODEL["seed"])
    ap.add_argument("--lr", type=float, default=MODEL["lr"])
    ap.add_argument("--cw-power", type=float, default=0.5)
    ap.add_argument("--far-budget", type=float, default=0.01,
                    help="dev false-alarm rate used to pick the alarm threshold")
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)

    d, meta = load_splits()
    Xtr, Xdv, Xte = d["X_train"], d["X_dev"], d["X_test"]
    ytr_c, ydv_c, yte_c = d["y_cls_train"], d["y_cls_dev"], d["y_cls_test"]
    ytr_r, ydv_r, yte_r = d["y_reg_train"], d["y_reg_dev"], d["y_reg_test"]

    mean, std = fit_normaliser(Xtr)
    Xtr_n, Xdv_n, Xte_n = (apply_norm(a, mean, std) for a in (Xtr, Xdv, Xte))

    r_mean, r_std = float(ytr_r.mean()), float(ytr_r.std()) or 1.0
    ytr_rn = ((ytr_r - r_mean) / r_std).astype(np.float32)
    ydv_rn = ((ydv_r - r_mean) / r_std).astype(np.float32)

    cw = class_weights(ytr_c, args.cw_power)
    print(f"[train] class weights   : { {CLASS_NAMES[k]: round(v,2) for k,v in cw.items()} }")
    print(f"[train] train/dev/test  : {len(Xtr)} / {len(Xdv)} / {len(Xte)}")

    model = build_model(Xtr.shape[1], Xtr.shape[2])
    n_par = model.count_params()
    print(f"[train] parameters      : {n_par:,}  (~{n_par/1024:.1f} KB as int8)")

    w_vec = tf.constant([cw[i] for i in range(N_CLASSES)], dtype=tf.float32)

    def weighted_sparse_ce(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return ce * tf.gather(w_vec, y_true)

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss={"health": weighted_sparse_ce, "wti": keras.losses.Huber(delta=1.0)},
        loss_weights=MODEL["loss_weights"],
        metrics={"health": ["accuracy"], "wti": ["mae"]},
    )

    f1cb = DevSelector(Xdv_n, ydv_c, patience=MODEL["patience"])
    cbs = [
        f1cb,
        keras.callbacks.ReduceLROnPlateau(monitor="val_wti_mae", mode="min",
                                          factor=0.5, patience=8,
                                          min_lr=1e-5, verbose=0),
    ]

    hist = model.fit(
        Xtr_n, {"health": ytr_c, "wti": ytr_rn},
        validation_data=(Xdv_n, {"health": ydv_c, "wti": ydv_rn}),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=cbs, verbose=0,
    )
    print(f"[train] epochs run      : {len(hist.history['loss'])}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACT_DIR / "model_fp32.keras")
    np.savez(ARTIFACT_DIR / "norm.npz", mean=mean, std=std,
             r_mean=np.float32(r_mean), r_std=np.float32(r_std))

    # ------------------------------------ alarm operating point (dev)
    # The 3-class argmax is dominated by persistence (see README).  The
    # actionable output is a single ALARM bit:
    #       risk = P(WARNING) + P(CRITICAL) within the horizon
    #
    # The threshold is chosen on the DEV split at a fixed FALSE-ALARM BUDGET
    # rather than at max-F1.  Max-F1 thresholds are tuned to the dev class
    # balance, and because the splits are chronological the fault rate drifts
    # between them -- a max-F1 threshold simply does not transfer.  A false
    # alarm budget is both stable under that drift and the way an operator
    # actually specifies the requirement ("no more than ~1 nuisance alarm
    # per N readings").
    pdv = model.predict(Xdv_n, verbose=0)[0]
    risk_dv = pdv[:, 1] + pdv[:, 2]
    alarm_dv = (ydv_c > 0).astype(int)
    neg = risk_dv[alarm_dv == 0]
    best_thr = float(np.quantile(neg, 1.0 - args.far_budget)) if len(neg) else 0.5
    best_thr = float(np.clip(best_thr, 0.01, 0.999))
    best_f1 = alarm_f1(ydv_c, risk_dv, best_thr)
    dev_rec = float((risk_dv[alarm_dv == 1] >= best_thr).mean()) if alarm_dv.any() else 0.0
    print(f"[train] alarm threshold : {best_thr:.3f}  "
          f"(dev FAR budget {args.far_budget*100:.1f} %  ->  dev recall "
          f"{dev_rec*100:.1f} %, dev alarm-F1 {best_f1:.3f})")

    # ------------------------------------------------ test-set snapshot
    pc, pr = model.predict(Xte_n, verbose=0)
    pred = pc.argmax(1)
    acc = float((pred == yte_c).mean())
    f1 = macro_f1(yte_c, pred)
    wti_pred = pr.ravel() * r_std + r_mean
    mae = float(np.abs(wti_pred - yte_r).mean())

    yte_now = d["y_now_test"]
    pers = yte_now                                   # persistence baseline
    pers_acc = float((pers == yte_c).mean())
    pers_f1 = macro_f1(yte_c, pers)

    ew = (yte_now == 0) & (yte_c > 0)                # genuine early warnings
    ew_model = float((pred[ew] > 0).mean()) if ew.any() else float("nan")

    print("\n[train] ================ TEST snapshot ================")
    print(f"[train] accuracy     model {acc*100:6.2f} %   | persistence {pers_acc*100:6.2f} %")
    print(f"[train] macro-F1     model {f1:6.4f}     | persistence {pers_f1:6.4f}")
    print(f"[train] early-warning recall (n={int(ew.sum())}): "
          f"model {ew_model*100:5.1f} %  | persistence 0.0 % (by definition)")
    print(f"[train] WTI MAE      {mae:6.2f} degC   (target std {yte_r.std():.2f})")
    # alarm view on test
    risk_te = pc[:, 1] + pc[:, 2]
    alarm_te = (yte_c > 0).astype(int)
    pred_a = (risk_te >= best_thr).astype(int)
    tp = int(((pred_a == 1) & (alarm_te == 1)).sum())
    fp = int(((pred_a == 1) & (alarm_te == 0)).sum())
    fn = int(((pred_a == 0) & (alarm_te == 1)).sum())
    tn = int(((pred_a == 0) & (alarm_te == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1a = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    print(f"[train] ALARM head   precision {prec*100:5.1f} %  recall {rec*100:5.1f} %  "
          f"F1 {f1a:.3f}  false-alarm {far*100:4.1f} %")
    print("[train] ===================================================")

    (ARTIFACT_DIR / "operating_point.json").write_text(json.dumps({
        "alarm_threshold": best_thr,
        "far_budget": args.far_budget,
        "dev_alarm_f1": best_f1,
        "dev_recall": dev_rec,
        "test": {"precision": prec, "recall": rec, "f1": f1a,
                 "false_alarm_rate": far, "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "early_warning_recall": ew_model,
        "wti_mae": mae,
        "macro_f1": f1, "accuracy": acc,
        "persistence": {"accuracy": pers_acc, "macro_f1": pers_f1},
    }, indent=2))

    hist.history["val_alarm_f1"] = f1cb.history
    (ARTIFACT_DIR / "train_history.json").write_text(json.dumps(
        {k: [float(x) for x in v] for k, v in hist.history.items()}, indent=2))
    print(f"[train] saved -> {ARTIFACT_DIR/'model_fp32.keras'}")


if __name__ == "__main__":
    main()
