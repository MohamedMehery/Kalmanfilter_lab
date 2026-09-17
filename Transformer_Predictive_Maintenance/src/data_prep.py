"""
Data loading, cleaning, feature engineering and train/dev/test splitting
for the transformer predictive-maintenance pipeline.

Run directly to (re)generate the processed CSV splits:

    python src/data_prep.py

Source data: Kaggle "Distributed Transformer Monitoring" dataset
(Overview.csv + CurrentVoltage.csv), collected via IoT devices every
~15 minutes from 2019-06-25 to 2020-04-14.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ALARM_COLS,
    BASE_FEATURE_COLS,
    BLOCK_SIZE,
    CURRENT_VOLTAGE_CSV,
    CURRENT_VOLTAGE_SENSOR_COLS,
    DATA_PROCESSED_DIR,
    DEV_FRAC,
    N_STRATA,
    OVERVIEW_CSV,
    OVERVIEW_SENSOR_COLS,
    PREDICTION_HORIZON,
    RANDOM_SEED,
    ROLLING_SOURCE_COLS,
    ROLLING_STATS,
    ROLLING_WINDOW,
    TARGET_COL,
    TEST_FRAC,
    TIMESTAMP_COL,
    TRAIN_FRAC,
)


def load_raw() -> pd.DataFrame:
    """Load, merge and de-duplicate the two raw source tables."""
    overview = pd.read_csv(OVERVIEW_CSV)
    curr_volt = pd.read_csv(CURRENT_VOLTAGE_CSV)

    overview[TIMESTAMP_COL] = pd.to_datetime(overview[TIMESTAMP_COL])
    curr_volt[TIMESTAMP_COL] = pd.to_datetime(curr_volt[TIMESTAMP_COL])

    merged = pd.merge(overview, curr_volt, on=TIMESTAMP_COL, how="inner")
    merged = (
        merged.sort_values(TIMESTAMP_COL)
        .drop_duplicates(subset=TIMESTAMP_COL)
        .reset_index(drop=True)
    )
    return merged


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal rolling statistics (only look-back, no leakage from the future)."""
    df = df.copy()
    for col in ROLLING_SOURCE_COLS:
        roll = df[col].rolling(window=ROLLING_WINDOW, min_periods=1)
        if "mean" in ROLLING_STATS:
            df[f"{col}_roll_mean"] = roll.mean()
        if "std" in ROLLING_STATS:
            df[f"{col}_roll_std"] = roll.std().fillna(0.0)
        df[f"{col}_delta"] = df[col].diff().fillna(0.0)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary predictive-maintenance target: will ANY alarm/trip flag
    (OTI_A, OTI_T, MOG_A) be active within the next PREDICTION_HORIZON
    samples (~1 hour at the dataset's ~15 min cadence)?
    """
    df = df.copy()
    any_alarm_now = (df[ALARM_COLS].sum(axis=1) > 0).astype(int)

    # Look ahead PREDICTION_HORIZON steps (rolling max over the future window).
    future_max = (
        any_alarm_now[::-1]
        .rolling(window=PREDICTION_HORIZON + 1, min_periods=1)
        .max()[::-1]
    )
    df[TARGET_COL] = future_max.astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    engineered = [c for c in df.columns if c.endswith(("_roll_mean", "_roll_std", "_delta"))]
    return BASE_FEATURE_COLS + engineered


def stratified_block_split(
    df: pd.DataFrame,
    block_size: int = BLOCK_SIZE,
    n_strata: int = N_STRATA,
    train_frac: float = TRAIN_FRAC,
    dev_frac: float = DEV_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Split the (already time-ordered) dataframe into train/dev/test while:
      1. keeping each contiguous BLOCK of `block_size` consecutive samples
         together (so we never chop a short-term temporal pattern in half
         between splits), and
      2. stratifying blocks by their positive-label rate so all three
         splits see a realistic mix of "quiet" and "alarm-heavy" periods
         instead of a plain chronological split, which for this dataset
         would put (almost) all positive labels in the training period and
         leave dev/test with virtually no positives (see config.py).

    Adds a `split` column with values {"train", "dev", "test"}.
    """
    assert abs(train_frac + dev_frac + test_frac - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    n = len(df)
    n_blocks = int(np.ceil(n / block_size))

    block_ids = np.repeat(np.arange(n_blocks), block_size)[:n]
    df = df.copy()
    df["_block"] = block_ids

    block_pos_rate = df.groupby("_block")[TARGET_COL].mean()
    # Quantile-bucket blocks by their positive rate (handle many zero-rate
    # blocks by using rank-based qcut with duplicate-safe bucketing).
    try:
        strata = pd.qcut(block_pos_rate.rank(method="first"), q=n_strata, labels=False)
    except ValueError:
        strata = pd.Series(0, index=block_pos_rate.index)

    split_assignment = {}
    for stratum in sorted(strata.unique()):
        stratum_blocks = strata[strata == stratum].index.to_numpy().copy()
        rng.shuffle(stratum_blocks)
        n_b = len(stratum_blocks)
        n_train = int(round(n_b * train_frac))
        n_dev = int(round(n_b * dev_frac))
        train_blocks = stratum_blocks[:n_train]
        dev_blocks = stratum_blocks[n_train:n_train + n_dev]
        test_blocks = stratum_blocks[n_train + n_dev:]
        for b in train_blocks:
            split_assignment[b] = "train"
        for b in dev_blocks:
            split_assignment[b] = "dev"
        for b in test_blocks:
            split_assignment[b] = "test"

    df["split"] = df["_block"].map(split_assignment)
    df = df.drop(columns=["_block"])
    return df


def build_dataset() -> pd.DataFrame:
    df = load_raw()
    df = add_rolling_features(df)
    df = add_target(df)
    df = stratified_block_split(df)
    return df


def main() -> None:
    df = build_dataset()
    feature_cols = get_feature_columns(df)

    out_cols = [TIMESTAMP_COL] + feature_cols + ALARM_COLS + [TARGET_COL, "split"]
    df_out = df[out_cols]

    full_path = DATA_PROCESSED_DIR / "transformer_dataset.csv"
    df_out.to_csv(full_path, index=False)
    print(f"Saved full processed dataset -> {full_path} ({len(df_out)} rows)")

    for split_name in ("train", "dev", "test"):
        split_df = df_out[df_out["split"] == split_name].drop(columns=["split"])
        path = DATA_PROCESSED_DIR / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        pos_rate = split_df[TARGET_COL].mean()
        print(f"{split_name:>5}: {len(split_df):6d} rows | positive rate = {pos_rate:.3%} -> {path}")

    with open(DATA_PROCESSED_DIR / "feature_columns.txt", "w") as f:
        f.write("\n".join(feature_cols))
    print(f"Saved {len(feature_cols)} feature column names -> feature_columns.txt")


if __name__ == "__main__":
    main()
