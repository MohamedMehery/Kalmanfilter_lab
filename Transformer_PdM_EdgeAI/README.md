# Transformer Predictive Maintenance — Edge AI (ESP32)

A complete TinyML pipeline for power-transformer condition monitoring:
**data → train/dev/test → int8 quantisation → verified C firmware on an ESP32.**

The deployed model is **8,204 parameters / 17.5 KB** and predicts, two hours ahead:

1. **Health state** — `NORMAL` / `WARNING` / `CRITICAL`
2. **An alarm bit** — the actual decision the node acts on
3. **Winding temperature (WTI)** — a numeric forecast in °C

```bash
./run_all.sh                 # synthetic physics-based data, ~4 min end to end
./run_all.sh your_data.csv   # your own transformer log
```

Or open [`05_notebooks/Transformer_PdM_Colab.ipynb`](05_notebooks/Transformer_PdM_Colab.ipynb) in Colab.

---

## Read this before trusting any accuracy number

A transformer's health state is **thermally slow**. Over a 2-hour horizon it
almost never changes. So the trivial rule *"the state in 2 hours = the state
right now"* — the **persistence baseline** — already scores:

| | accuracy | macro-F1 |
|---|---|---|
| **Persistence (predict no change)** | **95.45 %** | **0.631** |
| This CNN (float32) | 91.91 % | 0.497 |

**On raw 3-class accuracy, the model loses to doing nothing.** Reporting
"92 % accurate" here would be meaningless. That's the honest headline, and it
is why this project is evaluated on the two things persistence *cannot* do:

| metric | persistence | **this model** |
|---|---|---|
| **Early-warning recall** — looks healthy now, degrades within 2 h (236 cases) | **0 %** (impossible by construction) | **33.1 %** |
| **WTI forecast MAE** | 6.25 °C (carry-forward) | **2.66 °C** — 57 % better |

A gradient-boosting model with 300 trees on the same windows also reaches
macro-F1 ≈ 0.507, confirming the ceiling is a property of the task, not of the
8 K-parameter budget.

### The deployed decision: an alarm bit

Rather than argmax over three classes, the firmware thresholds
`risk = P(WARNING) + P(CRITICAL)`:

| | precision | recall | false-alarm rate |
|---|---|---|---|
| float32 | 65.4 % | 50.8 % | 2.1 % |
| **int8 (on device)** | **65.5 %** | **50.6 %** | **2.1 %** |

The threshold (0.913) is picked on the **dev split at a 1 % false-alarm
budget** — never on test. Max-F1 thresholds were tried first and did not
transfer (dev 0.87 → test 0.53): because the splits are chronological, the
fault rate drifts between them. A false-alarm budget is stable under that
drift and is how an operator actually specifies the requirement.

**Interpretation:** roughly half of all degradation events are caught 2 hours
early, and about 1 in 3 alarms is a false positive. For scheduling an
inspection that is useful; as an automatic trip signal it is not, which is why
the firmware only actuates the relay on a predicted `CRITICAL`.

---

## Layout

```
Transformer_PdM_EdgeAI/
├── config.py                  all tunables: schema, thresholds, hyper-params
├── run_all.sh                 one-command pipeline (fail-fast)
│
├── 01_data/
│   ├── make_synthetic.py      IEEE C57.91 thermal model + injected faults
│   └── preprocess.py          ingest → features → labels → chronological split
├── 02_training/train.py       multi-head 1-D CNN
├── 03_evaluation/evaluate.py  metrics vs baselines → report.md + plots/
├── 04_deployment_esp32/
│   ├── include/
│   │   ├── pdm_features.h            on-device feature engineering
│   │   ├── transformer_pdm_model.h   int8 model as a C array (generated)
│   │   └── replay_data.h             test rows + expected outputs (generated)
│   ├── src/main.cpp           ESP32 firmware
│   ├── src/sim_main.cpp       host build of the same feature code
│   ├── scripts/
│   │   ├── convert_tflite.py     quantise + verify + emit header
│   │   ├── test_parity.py        C features == Python features
│   │   └── verify_pipeline.py    full edge path == host decisions
│   └── platformio.ini, diagram.json, wokwi.toml
└── 05_notebooks/              Colab notebook
```

---

## Using your own data

Point the loader at your CSV — headers are mapped automatically:

```bash
python 01_data/preprocess.py --csv my_transformer_log.csv
```

`COLUMN_ALIASES` in `config.py` resolves the usual variants (`OTI`, `OilTemp`
and `Top-oil` all mean the same thing). Canonical fields:

| field | meaning | typical sensor |
|---|---|---|
| `VL1/2/3` | phase-neutral voltage | ZMPT101B / PT |
| `IL1/2/3` | phase current | SCT-013 / CT |
| `OTI` | top-oil temperature | PT100 |
| `WTI` | winding hot-spot temperature | PT100 |
| `ATI` | ambient temperature | DS18B20 |
| `OLI` | oil level % | float sender |

Missing columns are estimated and **reported in the log** — check that output,
because an estimated `ATI` weakens the thermal features considerably. Set the
transformer's rating with `--i-rated <amps>` if it is known; otherwise it is
inferred from the 99th percentile of load current.

### The synthetic generator

Used when no CSV is supplied. Not random noise — it integrates the IEEE
Std C57.91 clause-7 exponential thermal model:

```
Δθ_oil_ss = Δθ_or · ((K²R + 1)/(R + 1))^n        top-oil steady-state rise
Δθ_hs_ss  = Δθ_hr · K^(2m)                        hot-spot rise
Δθ(t+dt)  = Δθ(t) + (Δθ_ss − Δθ(t))(1 − e^(−dt/τ))
```

with `K` = per-unit load, `R = 5`, `n = m = 0.8` (ONAF), `τ_oil = 3 h`. Six
fault types are injected — cooling loss, oil leak, overload, voltage/current
unbalance, sensor spikes — each followed by a **maintenance recovery**, without
which degradation accumulates monotonically and every later sample becomes
`CRITICAL`. Resulting distribution: median WTI 67 °C, brief excursions to
~119 °C. All six labelling conditions fire.

---

## Design decisions worth knowing

**Labels are strictly in the future.** `y` is the worst state in `(t, t+H]`,
*excluding* the current sample. Including it would let the network score well
by re-applying threshold rules to a row it can already see — that is not
prediction.

**`sev_now` is an input feature, and it is not leakage.** It is the
deterministic threshold rule on the *current* sample: a pure function of
sensors already in the window, while the label lives strictly in the future.
Supplying it frees the 8 K-parameter model from rediscovering the standards
thresholds so it can spend capacity on the hard part — whether the state is
about to *escalate*. On the MCU it costs six comparisons.

**Chronological splits with a purge gap.** 70/15/15 by time, with a
`WINDOW + HORIZON` gap between segments so no sliding window straddles a
boundary. Shuffling a time series before splitting leaks neighbouring windows
between train and test and inflates every metric.

**Normalisation statistics come from train only**, and are exported to the C
header so the device applies exactly the same transform.

**No LSTM/GRU.** They lower to TFLM ops that are unsupported or arena-hungry.
A stacked 1-D CNN captures short-horizon dynamics far more cheaply.

**Model selection on dev alarm-F1.** Not `val_loss` (class weights make it
explode — an early version restored epoch-1 weights because of this) and not
accuracy (persistence-dominated).

**Square-root class weights.** Fully balanced weights reached ~50× and made
the model cry `CRITICAL` everywhere, destroying precision.

---

## Quantisation

Full-integer int8, inputs and outputs included, so there are no float
conversion ops at the graph boundary.

| | float32 | int8 |
|---|---|---|
| WTI MAE | 2.660 °C | **2.652 °C** |
| alarm precision | 65.4 % | 65.5 % |
| size | ~33 KB | **17.5 KB** |

Argmax agreement **99.22 %**, alarm-bit agreement **99.40 %**. Quantisation is
not assumed to be harmless — `convert_tflite.py` re-runs the entire test split
through the interpreter and fails loudly if it is not.

---

## Verifying before you flash

Train/serve skew is the classic silent TinyML failure: the firmware computes
features a little differently, the model still returns confident numbers, and
they are wrong. Two tests make that impossible to miss.

**1. Feature parity** — compiles `pdm_features.h` on the host and diffs all 23
features against the Python pipeline over 4,000 rows:

```
[parity] overall max diff = 1.507e-04
[parity] PASS — C and Python feature pipelines agree
```

This caught a real bug immediately: a missing `#include <stdlib.h>` made
`atof` implicitly return `int`, so `i_rated` silently became 1.0 and `load_pu`
was wrong by a factor of 1449. The harness now builds with `-Werror`.

**2. End-to-end edge verification** — runs the C feature pipeline into the int8
interpreter and compares against reference decisions baked into the firmware:

```
[verify] compared      : 221 inferences
[verify] max |d risk|  : 0.000005
[verify] max |d WTI|   : 0.0005 degC
[verify] class mismatch: 0     alarm mismatch: 0
[verify] PASS — edge path reproduces the host decisions exactly
```

**Warm-up is a real deployment property.** The 1-hour trend features need
`PDMF_LAG_1H` samples of history that a freshly booted MCU does not have, so
the first `WINDOW + LAG_1H = 20` samples produce features that differ from
training. `pdm_ready()` gates on this and the firmware refuses to raise alarms
until primed — at a 15-minute sample period that is **5 hours after boot**.
(This surfaced as a 3-LSB discrepancy on exactly two replay rows; it is
excluded deliberately, not papered over.)

---

## Firmware

```bash
cd 04_deployment_esp32
pio run -t upload && pio device monitor
```

`APP_MODE 0` (default) replays 240 bundled test rows so you can confirm the
board reproduces the host numbers. `APP_MODE 1` reads live sensors — implement
`read_sensors()` for your front-end.

Output is one JSON line per inference:

```json
{"t":45000,"wti":78.40,"oti":61.20,"load":0.812,"sev_now":0,
 "pred":"WARNING","risk":0.934,"alarm":true,"wti_fc":86.15,"us":4120}
```

Resources: **17.5 KB** flash for the model, a **24 KB** tensor arena
(`arena_used_bytes()` is printed at boot — shrink `PDM_TENSOR_ARENA` to match),
and ~4 ms per inference at 240 MHz. Inference runs every 15 minutes, so the
duty cycle is negligible and the node is deep-sleep friendly.

> **Sampling matters.** The model was trained on 15-minute *averages*. In LIVE
> mode, sample the AC channels at a few kHz for ≥10 cycles, compute true RMS,
> and average over the full period. Feeding instantaneous readings is the
> fastest way to make a good model look broken.

---

## Limitations

- **Headline results come from synthetic data.** The physics is standards-based
  and the faults are realistic, but synthetic data is smoother than reality.
  Expect degradation on real logs — rerun `./run_all.sh your.csv` and read the
  regenerated report rather than trusting the numbers above.
- **Labels are rule-derived, not maintenance records.** Ground truth is the
  IEEE/NEMA threshold rules, so the model learns *those rules' future state*,
  not verified failures. With real inspection outcomes the labelling in
  `preprocess.py::label_health` should be replaced.
- **~1 in 3 alarms is false.** Suitable for prioritising inspections; not a
  protection-grade trip signal.
- **No DGA.** Dissolved-gas analysis is the strongest transformer diagnostic
  and is absent — this node sees only electrical and thermal quantities.
- **Single transformer.** No cross-unit generalisation is demonstrated.

## References

- IEEE Std C57.91 — *Guide for Loading Mineral-Oil-Immersed Transformers*
  (clause 7 thermal model, Table 4 exponents)
- NEMA MG-1 — voltage unbalance definition
- TensorFlow Lite for Microcontrollers — int8 deployment

*Author: Mohamed Abdelnasser Mehery*
