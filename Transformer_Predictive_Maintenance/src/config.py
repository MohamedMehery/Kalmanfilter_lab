"""
Central configuration for the Transformer Predictive Maintenance pipeline.

Keeping every path / hyper-parameter in one place makes the pipeline
reproducible across local runs, Google Colab and the MCU export step.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for d in (DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

OVERVIEW_CSV = DATA_RAW_DIR / "Overview.csv"
CURRENT_VOLTAGE_CSV = DATA_RAW_DIR / "CurrentVoltage.csv"

# --------------------------------------------------------------------------
# Dataset schema (see Kaggle "Distributed Transformer Monitoring" dataset by
# Sreshta Putchala: https://www.kaggle.com/datasets/sreshta140/ai-transformer-monitoring)
# --------------------------------------------------------------------------
TIMESTAMP_COL = "DeviceTimeStamp"

# Raw sensor columns used as model inputs.
OVERVIEW_SENSOR_COLS = ["OTI", "ATI", "OLI"]          # WTI excluded: near-duplicate of label
CURRENT_VOLTAGE_SENSOR_COLS = [
    "VL1", "VL2", "VL3",
    "IL1", "IL2", "IL3",
    "VL12", "VL23", "VL31",
    "INUT",
]
BASE_FEATURE_COLS = OVERVIEW_SENSOR_COLS + CURRENT_VOLTAGE_SENSOR_COLS

# Alarm / trip flags used to build the predictive-maintenance target.
ALARM_COLS = ["OTI_A", "OTI_T", "MOG_A"]

# Rolling-window feature engineering (causal, i.e. only past+current samples).
ROLLING_WINDOW = 4     # ~1 hour of history at the dataset's ~15 min cadence
ROLLING_STATS = ["mean", "std"]
ROLLING_SOURCE_COLS = ["OTI", "ATI", "IL1", "IL2", "IL3"]

# Predictive horizon: how many samples ahead we forecast the alarm state.
# At ~15 minutes/sample, horizon=4 means "will an alarm trigger within ~1h?"
PREDICTION_HORIZON = 4

TARGET_COL = "future_alarm"

# --------------------------------------------------------------------------
# Train / dev / test split
# --------------------------------------------------------------------------
# A naive chronological split is unusable here: the alarm flags in this
# public dataset are extremely non-stationary (e.g. MOG_A only fires during
# a few weeks in mid-2019, OTI_A/OTI_T are almost entirely absent after
# 2019, WTI flips from "always 0" to "always 1" once the sensor was
# recalibrated in 2020). A plain time-ordered 70/15/15 split therefore
# yields a dev/test partition with (close to) zero positive labels, which
# makes evaluation meaningless.
#
# Instead we split at the level of contiguous BLOCKS of consecutive
# samples (to preserve local temporal structure / avoid trivial leakage
# between neighbouring rows) and then stratify the *blocks* across
# train/dev/test by their positive-label rate, so all three partitions
# contain a realistic, comparable mix of normal and alarm conditions
# drawn from across the full ~10 months of operation.
BLOCK_SIZE = 96          # ~1 day of samples per block
N_STRATA = 5              # quantile buckets used to stratify blocks by risk
TRAIN_FRAC = 0.70
DEV_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Model export
# --------------------------------------------------------------------------
CLASSIFICATION_THRESHOLD = 0.5
