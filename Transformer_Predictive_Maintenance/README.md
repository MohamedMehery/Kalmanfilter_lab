# Transformer Predictive Maintenance — Training Pipeline (PC / Google Colab)

Predictive-maintenance ML/DL pipeline for a real distribution transformer,
built on the **[Distributed Transformer Monitoring](https://www.kaggle.com/datasets/sreshta140/ai-transformer-monitoring)**
dataset (Sreshta Putchala, Kaggle): IoT sensor readings collected every
~15 minutes from 2019-06-25 to 2020-04-14.

This folder is the **first stage** of the project: everything here runs on
a local PC or Google Colab. Once a model is trained and validated here, it
is exported for on-device inference in the sibling
[`../Transformer_MCU_Deployment/`](../Transformer_MCU_Deployment) folder
(Arduino / MCU firmware).

```
Transformer_Predictive_Maintenance/
├── data/
│   ├── raw/                    # Original Kaggle CSVs (Overview.csv, CurrentVoltage.csv, ...)
│   └── processed/               # Generated: train.csv / dev.csv / test.csv + feature list
├── notebooks/
│   └── 01_train_transformer_predictive_maintenance.ipynb   # Runs locally OR on Colab
├── src/
│   ├── config.py                 # All paths & hyper-parameters in one place
│   ├── data_prep.py               # Load, merge, feature-engineer, split
│   ├── train_baseline.py          # Classical ML (LogReg / RandomForest / GradientBoosting)
│   ├── train_dl_model.py          # Small Keras MLP + TFLite export
│   ├── export_to_mcu.py           # Export weights to plain-C header for MCU
│   └── export_test_vectors.py     # Export held-out samples for on-device self-test
├── models/                       # Generated: trained models + scalers
├── reports/                      # Generated: metrics JSON + figures
└── requirements.txt
```

## Why a *predictive* target (not just "detect the fault right now")?

The raw dataset ships three fault/alarm indicator columns in `Overview.csv`:
`OTI_A` (oil temperature alarm), `OTI_T` (oil temperature trip), and
`MOG_A` (magnetic oil gauge alarm). Rather than just classifying the
*current* row, the target used here is:

> **Will any of these alarms be active within the next ~1 hour
> (`PREDICTION_HORIZON` = 4 samples @ ~15 min/sample)?**

This is a genuinely predictive-maintenance framing: the model only sees
*past and current* sensor readings (temperatures, 3-phase voltages and
currents, causal rolling statistics) and must forecast an incoming fault
before it fully manifests.

## A non-obvious data-quality issue this pipeline had to handle

The alarm flags in this public dataset are **extremely non-stationary**
across the ~10 months of collection (e.g. `MOG_A` only fires during a few
weeks in mid-2019; `WTI` is essentially always 0 in 2019 and always 1 after
a sensor recalibration in early 2020). A plain chronological 70/15/15 split
therefore puts almost all positive labels in `train` and leaves `dev`/`test`
with close to **zero** positive examples — useless for evaluation.

To fix this, `src/data_prep.py` uses a **stratified block split**:
1. The (still time-ordered) data is cut into contiguous blocks of
   `BLOCK_SIZE` = 96 samples (~1 day), preserving local temporal
   structure within each block.
2. Blocks are bucketed by their positive-label rate into quantile strata.
3. Each stratum is independently split 70/15/15 across train/dev/test.

The result: all three splits contain a comparable, realistic mix of quiet
and alarm-heavy periods drawn from across the full monitoring window
(see the printed positive rates when you run `data_prep.py`).

## Quickstart — local

```bash
cd Transformer_Predictive_Maintenance
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/data_prep.py          # -> data/processed/{train,dev,test}.csv
python src/train_baseline.py     # -> models/baseline_best_model.joblib
python src/train_dl_model.py     # -> models/dl_model.keras + .tflite (x2)
python src/export_to_mcu.py      # -> ../Transformer_MCU_Deployment/model/*.h
python src/export_test_vectors.py 20   # -> .../model/test_vectors.h
```

Or open `notebooks/01_train_transformer_predictive_maintenance.ipynb` and
run all cells (already executed once end-to-end and saved with outputs so
you can preview results without re-running anything).

## Quickstart — Google Colab

1. Upload this whole `Transformer_Predictive_Maintenance/` folder to
   Colab (or `git clone` the repo inside a Colab cell).
2. Open `notebooks/01_train_transformer_predictive_maintenance.ipynb` in
   Colab.
3. Run the first two cells (commented `pip install` / Kaggle-download
   snippets) to install dependencies and, if you don't already have the
   CSVs locally, pull them straight from Kaggle with your own
   `kaggle.json` API token.
4. Run all remaining cells — they are identical to the local pipeline.

## Results (held-out test split)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Gradient Boosting (best classical baseline) | 0.989 | 0.945 | 0.997 | 0.975 |
| Tiny MLP (Dense 28→16→8→1, MCU-ready)        | 0.940 | 0.764 | 0.982 | 0.894 |

(Exact numbers are regenerated on every run into `reports/baseline_results.json`
and `reports/dl_results.json` — small variations are expected depending on
package versions / random seeds.)

The classical Gradient Boosting model is the more accurate predictor, but
the tiny MLP is the one that gets deployed to the MCU: it's ~200x smaller,
runs in integer/float arithmetic with no external ML library, and still
reaches a strong 0.98 ROC-AUC.

## Feature set (28 features)

Raw sensors: `OTI, ATI, OLI` (Overview.csv) + `VL1, VL2, VL3, IL1, IL2, IL3,
VL12, VL23, VL31, INUT` (CurrentVoltage.csv).

Engineered (causal, backward-looking only): rolling mean / std / delta
over the last 4 samples (~1 hour) for `OTI, ATI, IL1, IL2, IL3` — the
signals most directly tied to thermal and electrical stress.

`WTI` (winding temperature indicator) is intentionally **excluded** from
the feature set: it is a near-perfect proxy for the label (see the EDA
section in the notebook) and including it would make the task trivial and
non-representative of a real early-warning system.

## Next step

→ [`../Transformer_MCU_Deployment/`](../Transformer_MCU_Deployment) —
Arduino/PlatformIO firmware that loads the exported model and runs
predictive-maintenance inference directly on an MCU.
