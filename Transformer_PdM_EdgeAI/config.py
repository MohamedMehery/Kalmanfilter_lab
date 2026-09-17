"""
Central configuration for the Transformer Predictive-Maintenance Edge-AI pipeline.

Everything that you may want to tweak lives here so that the rest of the code
stays untouched when you swap in your own dataset.

Author : Mohamed Abdelnasser Mehery
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "01_data"
ARTIFACT_DIR = ROOT / "artifacts"
EVAL_DIR = ROOT / "03_evaluation"
DEPLOY_DIR = ROOT / "04_deployment_esp32"

RAW_CSV = DATA_DIR / "raw" / "transformer_raw.csv"
SPLIT_DIR = DATA_DIR / "splits"

for _d in (DATA_DIR / "raw", SPLIT_DIR, ARTIFACT_DIR, EVAL_DIR / "plots"):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset schema
# --------------------------------------------------------------------------
# These are the canonical column names used internally.  If your CSV uses
# different headers, map them here:  {"your_header": "canonical_name"}
#
# The default mapping matches the widely used IoT "transformer monitoring"
# dumps (Kaggle / IEEE dataport style).  Unknown columns are ignored.
COLUMN_ALIASES: dict[str, str] = {
    # timestamp
    "DeviceTimeStamp": "timestamp",
    "Date": "timestamp",
    "DateTime": "timestamp",
    "time": "timestamp",
    # voltages (phase to neutral)
    "VL1": "VL1", "VL2": "VL2", "VL3": "VL3",
    "V1": "VL1", "V2": "VL2", "V3": "VL3",
    # currents
    "IL1": "IL1", "IL2": "IL2", "IL3": "IL3",
    "I1": "IL1", "I2": "IL2", "I3": "IL3",
    # phase-to-phase voltages
    "VL12": "VL12", "VL23": "VL23", "VL31": "VL31",
    # neutral current
    "INUT": "INUT", "IN": "INUT",
    # temperatures / oil
    "OTI": "OTI",  # oil temperature indicator
    "WTI": "WTI",  # winding temperature indicator
    "ATI": "ATI",  # ambient temperature indicator
    "OLI": "OLI",  # oil level indicator
    "OilTemp": "OTI",
    "WindingTemp": "WTI",
    "AmbientTemp": "ATI",
    "Top-oil": "OTI",
    "Hot-spot": "WTI",
    # alarms / status flags found in some dumps
    "OTI_A": "OTI_A", "OTI_T": "OTI_T",
    "MOG_A": "MOG_A",
}

# Sensors that a cheap ESP32 node can realistically measure.
# Order matters — the C header and the firmware rely on it.
BASE_FEATURES: list[str] = [
    "VL1", "VL2", "VL3",      # 3x voltage      (ZMPT101B / PT)
    "IL1", "IL2", "IL3",      # 3x current      (SCT-013 / CT)
    "OTI",                    # oil temperature (PT100 / DS18B20)
    "WTI",                    # winding temp    (PT100)
    "ATI",                    # ambient temp    (DHT22 / DS18B20)
    "OLI",                    # oil level       (float / ultrasonic)
]

# --------------------------------------------------------------------------
# Windowing / sampling
# --------------------------------------------------------------------------
SAMPLE_PERIOD_MIN = 15      # native sampling period of the dataset (minutes)
WINDOW = 16                 # 16 samples  = 4 hours of history
HORIZON = 8                 # predict 8 samples = 2 hours ahead
#   Why 2 h?  The oil time constant is ~3 h, so a 1 h horizon is almost
#   fully determined by the present state (state changes in <2 % of
#   windows) and any model just learns "copy the current state".
#   2 h is long enough for the thermal trajectory to matter and short
#   enough to still be actionable for an operator.
STRIDE = 1                  # sliding-window stride when building sequences

# --------------------------------------------------------------------------
# Health-state labelling thresholds (IEEE C57.91 inspired)
# --------------------------------------------------------------------------
# Class 0 = NORMAL, 1 = WARNING, 2 = CRITICAL
CLASS_NAMES = ["NORMAL", "WARNING", "CRITICAL"]
N_CLASSES = 3

THRESH = {
    "wti_warn": 95.0,       # deg C  winding hot-spot warning
    "wti_crit": 110.0,      # deg C  winding hot-spot critical (IEEE C57.91 limit)
    "oti_warn": 75.0,       # deg C  top-oil warning
    "oti_crit": 90.0,       # deg C  top-oil critical
    "oli_warn": 30.0,       # %      low oil level warning
    "oli_crit": 20.0,       # %      low oil level critical
    "unbal_warn": 2.0,      # %      voltage unbalance warning (NEMA MG-1)
    "unbal_crit": 5.0,      # %      voltage unbalance critical
    "iunbal_warn": 10.0,    # %      current unbalance warning
    "iunbal_crit": 25.0,    # %      current unbalance critical
    "load_warn": 1.10,      # pu     overload warning
    "load_crit": 1.30,      # pu     overload critical
}

# --------------------------------------------------------------------------
# Split strategy
# --------------------------------------------------------------------------
# Chronological split — NEVER shuffle a time series before splitting,
# otherwise neighbouring windows leak between train and test.
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / dev(val) / test

# --------------------------------------------------------------------------
# Model hyper-parameters
# --------------------------------------------------------------------------
MODEL = {
    "conv_filters": 24,
    "conv_kernel": 3,
    "gru_units": 0,          # 0 = disabled (GRU is not TFLite-Micro friendly)
    "dense_units": 32,
    "dropout": 0.20,
    "l2": 1e-4,
    "lr": 2e-3,
    "epochs": 120,
    "batch_size": 64,
    "patience": 25,
    "loss_weights": {"health": 1.0, "wti": 0.30},
    "seed": 42,
}

# --------------------------------------------------------------------------
# Edge / quantisation
# --------------------------------------------------------------------------
EDGE = {
    "quantize": "int8",         # int8 full-integer quantisation
    "rep_samples": 500,         # representative dataset size for calibration
    "tensor_arena_kb": 24,      # arena reserved in the firmware
    "target": "esp32",
}

# --------------------------------------------------------------------------
# Synthetic-data generator (used when no real CSV is present)
# --------------------------------------------------------------------------
SYNTH = {
    "n_days": 420,
    "seed": 7,
    "rating_kva": 1000.0,
    "v_nominal": 230.0,         # phase-to-neutral volts (secondary LV side)
    "i_rated": 1449.0,          # A  = 1000 kVA / (sqrt3 * 400 V)
    # IEEE C57.91 thermal model constants
    "dtheta_or": 45.0,          # top-oil rise over ambient at rated load (K)
    "dtheta_hr": 35.0,          # hot-spot rise over top-oil at rated load (K)
    "tau_oil_h": 3.0,           # oil time constant (hours)
    "tau_wind_h": 0.12,         # winding time constant (hours)
    "n_exp": 0.8,               # oil exponent (ONAF)
    "m_exp": 0.8,               # winding exponent
    "R_ratio": 5.0,             # load loss / no-load loss
    # fault injection
    "n_events": 85,
    "noise": {"v": 1.2, "i": 6.0, "t": 0.35, "oli": 0.4},
}

VERBOSE = int(os.environ.get("PDM_VERBOSE", "1"))
