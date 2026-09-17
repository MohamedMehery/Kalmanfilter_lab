"""
Data ingestion, feature engineering, labelling and chronological splitting.
===========================================================================

Design goals
------------
1. Accept *your* CSV whatever the headers are (see config.COLUMN_ALIASES).
2. Derive only features an ESP32 can actually compute in real time.
3. Label health states from physics/standards-based rules, not hand-waving.
4. Split strictly chronologically (train -> dev -> test) with a purge gap so
   no sliding window straddles a boundary => zero leakage.

Run:
    python 01_data/preprocess.py                 # uses 01_data/raw/transformer_raw.csv
    python 01_data/preprocess.py --csv mine.csv  # your own file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    BASE_FEATURES, COLUMN_ALIASES, HORIZON, RAW_CSV, SAMPLE_PERIOD_MIN,
    SPLIT_DIR, SPLIT_RATIOS, STRIDE, SYNTH, THRESH, WINDOW,
)

# Engineered features appended after the raw sensors.
DERIVED_FEATURES = [
    "V_avg",        # mean phase voltage
    "I_avg",        # mean phase current
    "V_unbal",      # % voltage unbalance  (NEMA MG-1 definition)
    "I_unbal",      # % current unbalance
    "load_pu",      # per-unit loading
    "dT_oil_amb",   # top-oil rise over ambient   (cooling health)
    "dT_wind_oil",  # winding rise over top-oil   (winding health)
    "OTI_roc",      # oil temperature rate of change (degC / sample)
    "WTI_roc",      # winding temperature rate of change
    "WTI_trend_1h", # winding temperature change over the last hour
    "OTI_trend_1h", # oil temperature change over the last hour
    "sev_now",      # rule-based severity RIGHT NOW (0/1/2)  <-- see note
    "margin_wti",   # normalised distance to the WTI warning threshold
]

# Note on `sev_now`
# -----------------
# This is the deterministic threshold rule evaluated on the CURRENT sample.
# It is NOT leakage: it is a pure function of sensors already present in the
# window, and the label lives strictly in the future.  Handing it to the model
# means the network no longer has to spend its tiny capacity re-discovering
# the standards-based thresholds, and can concentrate on the only thing that
# is actually hard -- predicting whether the state is about to ESCALATE.
# On the MCU the same rule is six comparisons, so this costs nothing.
FEATURES = BASE_FEATURES + DERIVED_FEATURES


# --------------------------------------------------------------- loading
def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV and normalise its column names to the canonical schema."""
    df = pd.read_csv(path)
    # case-insensitive alias match
    lower = {c.lower().strip(): c for c in df.columns}
    rename: dict[str, str] = {}
    for alias, canon in COLUMN_ALIASES.items():
        hit = lower.get(alias.lower())
        if hit is not None and hit not in rename:
            rename[hit] = canon
    df = df.rename(columns=rename)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)

    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        print(f"[preprocess] WARNING missing columns {missing} -> filled by estimate")
    return df


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill any sensor the source file does not provide with a sane estimate."""
    n = len(df)
    if "ATI" not in df.columns:
        # ambient not logged -> approximate with the daily minimum of top-oil
        if "OTI" in df.columns:
            df["ATI"] = df["OTI"].rolling(96, min_periods=1).min() - 2.0
        else:
            df["ATI"] = 25.0
    if "OLI" not in df.columns:
        df["OLI"] = 80.0
    if "WTI" not in df.columns and "OTI" in df.columns:
        df["WTI"] = df["OTI"] + 12.0
    for c in BASE_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[BASE_FEATURES] = (
        df[BASE_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    assert len(df) == n
    return df


# ------------------------------------------------------- feature building
def add_features(df: pd.DataFrame, i_rated: float | None = None) -> pd.DataFrame:
    v = df[["VL1", "VL2", "VL3"]].to_numpy(float)
    i = df[["IL1", "IL2", "IL3"]].to_numpy(float)

    v_avg = v.mean(axis=1)
    i_avg = i.mean(axis=1)

    # NEMA MG-1: % unbalance = max deviation from average / average * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        v_unb = np.where(v_avg > 1e-6,
                         np.abs(v - v_avg[:, None]).max(axis=1) / v_avg * 100.0, 0.0)
        i_unb = np.where(i_avg > 1e-6,
                         np.abs(i - i_avg[:, None]).max(axis=1) / i_avg * 100.0, 0.0)

    if i_rated is None:
        # infer the rating as the 99th percentile of average current
        i_rated = float(np.percentile(i_avg, 99)) or 1.0
        i_rated = max(i_rated, 1e-6)

    df["V_avg"] = v_avg
    df["I_avg"] = i_avg
    df["V_unbal"] = np.clip(v_unb, 0, 100)
    df["I_unbal"] = np.clip(i_unb, 0, 200)
    df["load_pu"] = i_avg / i_rated
    df["dT_oil_amb"] = df["OTI"] - df["ATI"]
    df["dT_wind_oil"] = df["WTI"] - df["OTI"]
    df["OTI_roc"] = df["OTI"].diff().fillna(0.0)
    df["WTI_roc"] = df["WTI"].diff().fillna(0.0)
    # 1-hour trends: the stacked Conv1D has a receptive field of only a few
    # samples, so longer-span slopes are supplied explicitly.
    lag_1h = max(1, int(60 / SAMPLE_PERIOD_MIN))
    df["WTI_trend_1h"] = df["WTI"].diff(lag_1h).fillna(0.0)
    df["OTI_trend_1h"] = df["OTI"].diff(lag_1h).fillna(0.0)
    df["margin_wti"] = (df["WTI"] - THRESH["wti_warn"]) / 10.0
    df["sev_now"] = 0.0                      # filled by label_health()
    df.attrs["i_rated"] = i_rated
    return df


# ------------------------------------------------------------- labelling
def label_health(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based health state grounded in transformer-loading standards.

    Severity score = max over individual condition severities:
        0 = NORMAL, 1 = WARNING, 2 = CRITICAL

    Rules (IEEE C57.91 loading guide + NEMA MG-1 unbalance limits):
        * winding hot-spot temperature  (thermal ageing driver)
        * top-oil temperature
        * oil level                     (cooling / insulation risk)
        * voltage & current unbalance   (negative-sequence heating)
        * per-unit overload
    """
    t = THRESH
    sev = np.zeros(len(df), dtype=np.int8)

    def bump(mask_warn, mask_crit):
        nonlocal sev
        sev = np.maximum(sev, np.where(mask_crit, 2, np.where(mask_warn, 1, 0)))

    bump(df.WTI >= t["wti_warn"], df.WTI >= t["wti_crit"])
    bump(df.OTI >= t["oti_warn"], df.OTI >= t["oti_crit"])
    bump(df.OLI <= t["oli_warn"], df.OLI <= t["oli_crit"])
    bump(df.V_unbal >= t["unbal_warn"], df.V_unbal >= t["unbal_crit"])
    bump(df.I_unbal >= t["iunbal_warn"], df.I_unbal >= t["iunbal_crit"])
    bump(df.load_pu >= t["load_warn"], df.load_pu >= t["load_crit"])

    df["health"] = sev
    df["sev_now"] = sev.astype(np.float32)   # same value, used as a model INPUT
    return df


def build_sequences(df: pd.DataFrame, features: list[str]):
    """
    Turn the table into sliding windows.

    X       : (N, WINDOW, F)  history of engineered features, samples t-W+1..t
    y_cls   : (N,)            worst health state strictly in the FUTURE window
                              (t, t+HORIZON].  The current sample is excluded
                              on purpose: including it would let the network
                              score well by simply re-applying the threshold
                              rules to the last row it can already see, which
                              is not prediction.
    y_reg   : (N,)            winding temperature exactly HORIZON steps ahead
    y_now   : (N,)            health state at t -- NOT a model input.  Kept so
                              the evaluation can build an honest "persistence"
                              baseline and isolate the early-warning subset.
    """
    X_cols = df[features].to_numpy(np.float32)
    health = df["health"].to_numpy(np.int8)
    wti = df["WTI"].to_numpy(np.float32)

    n = len(df)
    last = n - HORIZON
    idx = np.arange(WINDOW - 1, last, STRIDE)

    X = np.stack([X_cols[i - WINDOW + 1: i + 1] for i in idx]).astype(np.float32)
    # strictly-future label: worst health inside (t, t+HORIZON]
    y_cls = np.stack([health[i + 1: i + HORIZON + 1].max() for i in idx]).astype(np.int8)
    y_reg = wti[idx + HORIZON].astype(np.float32)
    y_now = health[idx].astype(np.int8)
    return X, y_cls, y_reg, y_now, idx


def chronological_split(n: int, ratios=SPLIT_RATIOS, purge: int = WINDOW + HORIZON):
    """Contiguous train/dev/test split with a purge gap between segments."""
    n_tr = int(n * ratios[0])
    n_dv = int(n * ratios[1])
    tr = np.arange(0, n_tr - purge)
    dv = np.arange(n_tr, n_tr + n_dv - purge)
    te = np.arange(n_tr + n_dv, n)
    return tr, dv, te


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(RAW_CSV))
    ap.add_argument("--i-rated", type=float, default=None,
                    help="rated phase current (A); inferred if omitted")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"[preprocess] {path} not found -> generating synthetic data first")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import make_synthetic
        make_synthetic.main()
        path = RAW_CSV

    df = load_csv(path)
    df = ensure_columns(df)
    i_rated = args.i_rated
    if i_rated is None and "_load_pu" in df.columns:
        i_rated = SYNTH["i_rated"]        # exact value when using synthetic data
    df = add_features(df, i_rated)
    df = label_health(df)

    X, y_cls, y_reg, y_now, idx = build_sequences(df, FEATURES)
    tr, dv, te = chronological_split(len(X))

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SPLIT_DIR / "dataset.npz",
        X_train=X[tr], y_cls_train=y_cls[tr], y_reg_train=y_reg[tr], y_now_train=y_now[tr],
        X_dev=X[dv], y_cls_dev=y_cls[dv], y_reg_dev=y_reg[dv], y_now_dev=y_now[dv],
        X_test=X[te], y_cls_test=y_cls[te], y_reg_test=y_reg[te], y_now_test=y_now[te],
    )

    meta = {
        "features": FEATURES,
        "n_features": len(FEATURES),
        "window": WINDOW,
        "horizon": HORIZON,
        "i_rated": float(df.attrs.get("i_rated", 0.0)),
        "rows_raw": int(len(df)),
        "sequences": {"train": int(len(tr)), "dev": int(len(dv)), "test": int(len(te))},
        "class_balance": {
            split: np.bincount(y[: ], minlength=3).tolist()
            for split, y in (("train", y_cls[tr]), ("dev", y_cls[dv]), ("test", y_cls[te]))
        },
        "wti_stats": {"mean": float(y_reg.mean()), "std": float(y_reg.std())},
        "early_warning": {
            # windows where the unit looks fine NOW but degrades within the
            # horizon -- the cases that actually justify an ML model
            split: int(((yn == 0) & (yc > 0)).sum())
            for split, yn, yc in (
                ("train", y_now[tr], y_cls[tr]),
                ("dev", y_now[dv], y_cls[dv]),
                ("test", y_now[te], y_cls[te]),
            )
        },
        "persistence_acc": {
            split: float((yn == yc).mean())
            for split, yn, yc in (
                ("train", y_now[tr], y_cls[tr]),
                ("dev", y_now[dv], y_cls[dv]),
                ("test", y_now[te], y_cls[te]),
            )
        },
    }
    (SPLIT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    df.to_csv(SPLIT_DIR / "processed_full.csv", index=False)

    print("[preprocess] features :", len(FEATURES))
    print("[preprocess] windows  :", X.shape)
    print("[preprocess] split    :", meta["sequences"])
    print("[preprocess] balance  :", meta["class_balance"])
    print(f"[preprocess] saved    : {SPLIT_DIR/'dataset.npz'}")


if __name__ == "__main__":
    main()
