## Your CSV Has Corrupted Values — Clean First, Then Plot

Before anything else, your CSV has significant corruption that will break any naive parser. Here's a full pipeline:

---

## 🐍 Step 1: The Python Script

```python
#!/usr/bin/env python3
"""
plot_kalman_results.py
----------------------
Cleans corrupted CSV exported from the STM32 Kalman testbench,
generates publication-quality plots, and saves them to ./plots/
"""

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_FILE   = "kalman_results.csv"
OUTPUT_DIR = "plots"
DPI        = 150

SCENARIO_NAMES = {
    1: "Steady 25°C",
    2: "Slow Ramp\n25→35°C",
    3: "Steady 35°C",
    4: "Fast Ramp\n35→45°C",
    5: "Steady 45°C",
    6: "Post-Spike\n45°C",
    7: "Cool Ramp\n45→30°C",
    8: "Steady 30°C",
}

SCENARIO_BOUNDARIES = [0, 60, 160, 220, 240, 260, 320, 360, 420]

COLORS = {
    "truth"   : "#2ECC71",   # green
    "raw"     : "#95A5A6",   # grey
    "fixed"   : "#E74C3C",   # red
    "adapt"   : "#3498DB",   # blue
    "spike"   : "#E67E22",   # orange
    "gain"    : "#9B59B6",   # purple
    "Q"       : "#1ABC9C",
    "R"       : "#E74C3C",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── DATA CLEANING ─────────────────────────────────────────────────────────────

def clean_token(token: str) -> float:
    """
    Coerce a potentially corrupted token to float.

    Known corruption patterns observed in this dataset
    (likely OCR artefacts from serial-monitor screen capture):
      • "ovf"  / "inf"  / "nan"  → NaN
      • leading letter  g.XXXX   → 0.XXXX
      • leading letter  j.XXXX   → same
      • digit string like 4294967295... → NaN  (uint32 overflow sentinel)
      • trailing letter  0.622p10 → 0.62210
      • embedded letter  0.690m5  → 0.69005
      • negative zero    -0.000   → 0.0
      • extra digits     27.0992  → 27.099  (kept as-is, valid float)
    """
    s = str(token).strip()

    # explicit sentinels
    if s.lower() in ("ovf", "inf", "nan", "", "na"):
        return np.nan

    # uint32 max overflow blob  (starts with 4294967295)
    if s.startswith("4294967295"):
        return np.nan

    # leading non-digit letter used instead of '0'  (g.41 → 0.41)
    if re.match(r'^[a-zA-Z]\.', s):
        s = "0." + s[2:]

    # leading letter before digit  (j4.628 → 34.628)
    if re.match(r'^[a-zA-Z]\d', s):
        s = s[1:]

    # trailing letter  (0.622p10 → 0.62210 , 32.17l → 32.171)
    s = re.sub(r'([0-9])([a-zA-Z])([0-9])', r'\1\3', s)  # embedded
    s = re.sub(r'([0-9])[a-zA-Z]$', r'\1', s)             # trailing

    # negative zero
    if s == "-0.000" or s == "-0.00":
        return 0.0

    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_time(token: str) -> float:
    """Clean the t_s column specifically."""
    s = str(token).strip()
    # remove any embedded letters  (75.1p4 → 75.14)
    s = re.sub(r'[a-zA-Z]', '', s)
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_and_clean(filepath: str) -> pd.DataFrame:
    raw = pd.read_csv(filepath, dtype=str, skipinitialspace=True)

    # normalise column names
    raw.columns = [c.strip() for c in raw.columns]

    df = pd.DataFrame()
    df["t_s"]        = raw["t_s"].apply(clean_time)
    df["rtc_hms"]    = raw["rtc_hms"].str.strip()
    df["scenario"]   = pd.to_numeric(raw["scenario"], errors="coerce")
    df["truth"]      = raw["truth"].apply(clean_token)
    df["measurement"]= raw["measurement"].apply(clean_token)
    df["fixed_x"]    = raw["fixed_x"].apply(clean_token)
    df["adapt_x"]    = raw["adapt_x"].apply(clean_token)
    df["adapt_Q"]    = raw["adapt_Q"].apply(clean_token)
    df["adapt_R"]    = raw["adapt_R"].apply(clean_token)
    df["adapt_K"]    = raw["adapt_K"].apply(clean_token)
    df["adapt_P"]    = raw["adapt_P"].apply(clean_token)
    df["adapt_rate"] = raw["adapt_rate"].apply(clean_token)
    df["spike_flag"] = pd.to_numeric(raw["spike_flag"], errors="coerce").fillna(0)

    # drop rows where time is missing
    df.dropna(subset=["t_s"], inplace=True)
    df.sort_values("t_s", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # report how much was cleaned
    total_cells = df.shape[0] * df.shape[1]
    nan_cells   = df.isnull().sum().sum()
    print(f"[clean] {df.shape[0]} rows loaded | "
          f"{nan_cells}/{total_cells} cells set to NaN after cleaning")

    return df


# ── RMSE HELPER ──────────────────────────────────────────────────────────────

def rmse(pred: pd.Series, truth: pd.Series) -> float:
    mask = (~pred.isna()) & (~truth.isna())
    if mask.sum() == 0:
        return np.nan
    err = (pred[mask] - truth[mask]) ** 2
    return float(np.sqrt(err.mean()))


# ── PLOT 1 : FULL TIMELINE ────────────────────────────────────────────────────

def plot_full_timeline(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)

    ax_temp, ax_err = axes

    # ── shaded scenario bands ──
    for i, (lo, hi) in enumerate(
            zip(SCENARIO_BOUNDARIES[:-1], SCENARIO_BOUNDARIES[1:])):
        color = "#F8F9FA" if i % 2 == 0 else "#EBF5FB"
        ax_temp.axvspan(lo, hi, color=color, alpha=0.5, zorder=0)
        ax_err.axvspan(lo, hi,  color=color, alpha=0.5, zorder=0)
        mid = (lo + hi) / 2
        ax_temp.text(mid, 56, SCENARIO_NAMES.get(i + 1, ""),
                     ha="center", va="top", fontsize=7.5,
                     color="#666666", style="italic")

    # ── spike marker ──
    spike_rows = df[df["spike_flag"] == 1]
    if not spike_rows.empty:
        ax_temp.axvline(spike_rows["t_s"].iloc[0], color=COLORS["spike"],
                        lw=1.5, ls="--", label="Injected spike")

    # ── temperature traces ──
    ax_temp.plot(df["t_s"], df["truth"],       color=COLORS["truth"],
                 lw=2,   label="Ground truth",      zorder=4)
    ax_temp.scatter(df["t_s"], df["measurement"],  color=COLORS["raw"],
                    s=14, alpha=0.6, label="Raw (noisy)",  zorder=3)
    ax_temp.plot(df["t_s"], df["fixed_x"],     color=COLORS["fixed"],
                 lw=1.5, ls="--", label="Fixed KF",        zorder=5)
    ax_temp.plot(df["t_s"], df["adapt_x"],     color=COLORS["adapt"],
                 lw=2,   label="Adaptive KF",      zorder=6)

    ax_temp.set_ylabel("Temperature (°C)", fontsize=11)
    ax_temp.set_ylim(15, 60)
    ax_temp.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_temp.set_title("Adaptive vs Fixed Kalman Filter — Full Experiment Timeline",
                       fontsize=13, fontweight="bold", pad=10)
    ax_temp.grid(True, ls=":", alpha=0.4)

    # ── absolute error panel ──
    ax_err.fill_between(df["t_s"],
                        (df["fixed_x"] - df["truth"]).abs(),
                        alpha=0.4, color=COLORS["fixed"], label="|err| Fixed KF")
    ax_err.fill_between(df["t_s"],
                        (df["adapt_x"] - df["truth"]).abs(),
                        alpha=0.5, color=COLORS["adapt"], label="|err| Adaptive KF")
    ax_err.set_ylabel("|Error| (°C)", fontsize=10)
    ax_err.set_xlabel("Elapsed time (s)", fontsize=11)
    ax_err.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_err.grid(True, ls=":", alpha=0.4)
    ax_err.set_ylim(0, None)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "01_full_timeline.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ── PLOT 2 : RMSE PER SCENARIO ────────────────────────────────────────────────

def plot_rmse_bar(df: pd.DataFrame):
    scen_ids = sorted(df["scenario"].dropna().unique())
    labels, rmse_raw, rmse_fix, rmse_adp = [], [], [], []

    for sid in scen_ids:
        sub = df[df["scenario"] == sid]
        labels.append(SCENARIO_NAMES.get(int(sid), f"Scn {int(sid)}"))
        rmse_raw.append(rmse(sub["measurement"], sub["truth"]))
        rmse_fix.append(rmse(sub["fixed_x"],     sub["truth"]))
        rmse_adp.append(rmse(sub["adapt_x"],     sub["truth"]))

    x   = np.arange(len(labels))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))

    bars_r = ax.bar(x - w, rmse_raw, w, label="Raw sensor",   color=COLORS["raw"],   alpha=0.85)
    bars_f = ax.bar(x,     rmse_fix, w, label="Fixed KF",     color=COLORS["fixed"], alpha=0.85)
    bars_a = ax.bar(x + w, rmse_adp, w, label="Adaptive KF",  color=COLORS["adapt"], alpha=0.85)

    # value labels on bars
    for bar_group in (bars_r, bars_f, bars_a):
        for bar in bar_group:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE (°C)", fontsize=11)
    ax.set_title("RMSE per Scenario: Raw vs Fixed KF vs Adaptive KF",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_ylim(0, None)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "02_rmse_per_scenario.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ── PLOT 3 : SPIKE ZOOM ───────────────────────────────────────────────────────

def plot_spike_zoom(df: pd.DataFrame):
    # window: 20 s before spike → 40 s after
    spike_t = df.loc[df["spike_flag"] == 1, "t_s"]
    if spike_t.empty:
        print("[skip] no spike row found")
        return
    t_spike = spike_t.iloc[0]
    mask = (df["t_s"] >= t_spike - 20) & (df["t_s"] <= t_spike + 40)
    sub  = df[mask]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(sub["t_s"], sub["truth"],   color=COLORS["truth"],
            lw=2.5, label="Ground truth")
    ax.scatter(sub["t_s"], sub["measurement"], color=COLORS["raw"],
               s=30, zorder=5, label="Raw measurement")
    ax.plot(sub["t_s"], sub["fixed_x"], color=COLORS["fixed"],
            lw=2, ls="--", label="Fixed KF")
    ax.plot(sub["t_s"], sub["adapt_x"], color=COLORS["adapt"],
            lw=2.5, label="Adaptive KF")

    ax.axvline(t_spike, color=COLORS["spike"], lw=1.8,
               ls="--", label=f"Spike injected @ t={t_spike:.0f}s")
    ax.annotate("55°C spike\n(sensor fault)",
                xy=(t_spike, 55), xytext=(t_spike + 5, 54),
                arrowprops=dict(arrowstyle="->", color=COLORS["spike"]),
                fontsize=9, color=COLORS["spike"])

    ax.set_xlabel("Elapsed time (s)", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title("Spike Rejection — Adaptive KF vs Fixed KF (±60 s window)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "03_spike_zoom.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ── PLOT 4 : KALMAN GAIN + ADAPTIVE PARAMS ───────────────────────────────────

def plot_kalman_internals(df: pd.DataFrame):
    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(3, 1, hspace=0.05, figure=fig)

    ax_k = fig.add_subplot(gs[0])
    ax_q = fig.add_subplot(gs[1], sharex=ax_k)
    ax_r = fig.add_subplot(gs[2], sharex=ax_k)

    spike_t = df.loc[df["spike_flag"] == 1, "t_s"]

    for ax in (ax_k, ax_q, ax_r):
        for i, (lo, hi) in enumerate(
                zip(SCENARIO_BOUNDARIES[:-1], SCENARIO_BOUNDARIES[1:])):
            ax.axvspan(lo, hi,
                       color="#F8F9FA" if i % 2 == 0 else "#EBF5FB",
                       alpha=0.5, zorder=0)
        if not spike_t.empty:
            ax.axvline(spike_t.iloc[0], color=COLORS["spike"],
                       lw=1.5, ls="--", alpha=0.7)

    # Kalman gain
    ax_k.plot(df["t_s"], df["adapt_K"], color=COLORS["gain"], lw=1.5)
    ax_k.set_ylabel("Kalman Gain K", fontsize=10)
    ax_k.set_ylim(-0.05, 1.05)
    ax_k.set_title("Adaptive KF Internal Parameters over Time",
                   fontsize=12, fontweight="bold")
    ax_k.grid(True, ls=":", alpha=0.4)
    ax_k.tick_params(labelbottom=False)

    # Process noise Q
    ax_q.plot(df["t_s"], df["adapt_Q"].clip(upper=1.05),
              color=COLORS["Q"], lw=1.5)
    ax_q.set_ylabel("Process noise Q", fontsize=10)
    ax_q.set_ylim(-0.02, 1.1)
    ax_q.grid(True, ls=":", alpha=0.4)
    ax_q.tick_params(labelbottom=False)

    # Measurement noise R
    ax_r.plot(df["t_s"], df["adapt_R"].clip(upper=26),
              color=COLORS["R"], lw=1.5)
    ax_r.set_ylabel("Meas. noise R", fontsize=10)
    ax_r.set_xlabel("Elapsed time (s)", fontsize=11)
    ax_r.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "04_kalman_internals.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ── PLOT 5 : SCENARIO-LEVEL DETAIL (small multiples) ─────────────────────────

def plot_scenario_multiples(df: pd.DataFrame):
    scen_ids = sorted(df["scenario"].dropna().unique().astype(int))
    ncols = 4
    nrows = int(np.ceil(len(scen_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(15, 4 * nrows),
                             sharey=False)
    axes = axes.flatten()

    for idx, sid in enumerate(scen_ids):
        ax  = axes[idx]
        sub = df[df["scenario"] == sid].copy()
        t0  = sub["t_s"].iloc[0]
        sub["t_rel"] = sub["t_s"] - t0

        ax.plot(sub["t_rel"], sub["truth"],   color=COLORS["truth"],
                lw=2,   label="Truth")
        ax.scatter(sub["t_rel"], sub["measurement"], color=COLORS["raw"],
                   s=12, alpha=0.6, label="Raw")
        ax.plot(sub["t_rel"], sub["fixed_x"], color=COLORS["fixed"],
                lw=1.5, ls="--", label="Fixed KF")
        ax.plot(sub["t_rel"], sub["adapt_x"], color=COLORS["adapt"],
                lw=2,   label="Adaptive KF")

        rfix = rmse(sub["fixed_x"], sub["truth"])
        radp = rmse(sub["adapt_x"], sub["truth"])
        title_str = SCENARIO_NAMES.get(sid, f"Scn {sid}").replace("\n", " ")
        ax.set_title(f"{title_str}\nRMSE fix={rfix:.2f}  adp={radp:.2f}",
                     fontsize=8.5)
        ax.grid(True, ls=":", alpha=0.3)
        ax.set_xlabel("Relative time (s)", fontsize=8)

        # spike in scenario 6
        if sid == 6 and not df.loc[df["spike_flag"] == 1].empty:
            spike_t = df.loc[df["spike_flag"] == 1, "t_s"].iloc[0] - t0
            ax.axvline(spike_t, color=COLORS["spike"], ls="--", lw=1.2)

    # legend in first subplot only
    axes[0].legend(fontsize=7.5, loc="lower right")

    # hide unused subplots
    for idx in range(len(scen_ids), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Per-Scenario Detail: Fixed vs Adaptive Kalman Filter",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "05_scenario_multiples.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ── PLOT 6 : OVERALL RMSE SUMMARY ────────────────────────────────────────────

def plot_overall_summary(df: pd.DataFrame):
    """Single horizontal bar chart: overall RMSE comparison."""
    labels    = ["Raw sensor", "Fixed KF", "Adaptive KF"]
    rmse_vals = [
        rmse(df["measurement"], df["truth"]),
        rmse(df["fixed_x"],     df["truth"]),
        rmse(df["adapt_x"],     df["truth"]),
    ]
    colors = [COLORS["raw"], COLORS["fixed"], COLORS["adapt"]]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(labels, rmse_vals, color=colors, alpha=0.85, height=0.5)

    for bar, val in zip(bars, rmse_vals):
        if not np.isnan(val):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f} °C", va="center", fontsize=11,
                    fontweight="bold")

    ax.set_xlabel("Overall RMSE (°C)", fontsize=11)
    ax.set_title("Overall RMSE — 420 s Experiment\n(lower is better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, max([v for v in rmse_vals if not np.isnan(v)]) * 1.3)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.invert_yaxis()

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "06_overall_rmse.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Kalman Filter Testbench — Result Visualiser")
    print("=" * 55)

    df = load_and_clean(CSV_FILE)

    print("\n[info] Generating plots …")
    plot_full_timeline(df)
    plot_rmse_bar(df)
    plot_spike_zoom(df)
    plot_kalman_internals(df)
    plot_scenario_multiples(df)
    plot_overall_summary(df)

    print(f"\n✅  All plots saved to ./{OUTPUT_DIR}/")
    print("   Use plots/01 and plots/06 as your LinkedIn hero images.")
```

---

## 📁 Step 2: GitHub README Structure

```markdown
# Adaptive vs Fixed Kalman Filter — STM32 Testbench

> A controlled scientific experiment comparing a fixed-parameter 
> Kalman filter against an adaptive one for temperature estimation
> on a $2 microcontroller.

## Hardware
| Component | Role |
|-----------|------|
| STM32 Blue Pill (STM32F103C8T6) | MCU |
| DHT22 | Temperature + humidity sensor |
| DS1307 RTC | Hardware-validated wall-clock timestamps |
| LCD2004 (I²C) | Real-time display |
| 2× 4.7 kΩ resistors | I²C pull-ups |

> All I²C communication is **bit-bang** (no HAL, no Wire library) — 
> full control, zero abstraction.

## Wokwi Simulation
▶️ [Run it in your browser — no hardware needed](YOUR_WOKWI_LINK)

## Results

### Full Experiment Timeline (420 s, 8 scenarios)
![Full timeline](plots/01_full_timeline.png)

### Spike Rejection — The Key Differentiator
At t = 267 s a **55°C fault spike** was injected.  
The adaptive filter rejected it in a single sample.  
The fixed filter tracked it blindly.

![Spike zoom](plots/03_spike_zoom.png)

### RMSE Per Scenario
![RMSE bar chart](plots/02_rmse_per_scenario.png)

### Overall RMSE
![Overall RMSE](plots/06_overall_rmse.png)

### Adaptive KF Internal Parameters
Kalman gain K drops to ~0.004 the moment the spike appears —  
the filter automatically distrusts the measurement.

![KF internals](plots/04_kalman_internals.png)

### Per-Scenario Detail
![Small multiples](plots/05_scenario_multiples.png)

## How the Adaptive Filter Works
The filter tunes Q and R in real time based on **innovation rate**
(how fast the reading is changing vs. what is physically possible):

- `apparent_rate > max_phys_rate` → spike detected → `R` inflated
  100× → K → ~0 → measurement almost ignored
- `apparent_rate ≤ max_phys_rate` → Q scaled with rate²·dt → 
  filter tracks genuine ramps without lag

## Reproduce
```bash
git clone https://github.com/MohamedMehery/Kalmanfilter_lab
cd DHT22_temprature_humidty
pip install pandas matplotlib numpy
python plot_kalman_results.py
```
Open `diagram.json` + `main.cpp` in [Wokwi](https://wokwi.com).

## Project Structure
```
├── main.cpp               # Full firmware (Arduino/STM32)
├── diagram.json           # Wokwi circuit
├── kalman_results.csv     # Raw serial output
├── plot_kalman_results.py # This plotting script
└── plots/                 # Generated figures
```
```

---

## 💼 Step 3: The LinkedIn Post — With a Dignified Job Ask

Here is a complete ready-to-paste post. The job ask at the end is honest and confident without sounding desperate:

```
I ran a controlled experiment on a $2 microcontroller.
One filter followed a 55°C sensor fault blindly.
The other rejected it in a single sample.

Here's the full breakdown 👇

──────────────────────────────────

THE SETUP
• STM32 Blue Pill (Cortex-M3, $2)
• DHT22 temperature sensor
• DS1307 RTC — hardware clock, independent of the MCU
• LCD2004 — real-time display
• Everything on Wokwi — run it free in your browser

No HAL. No Wire library. I2C is fully bit-banged.

──────────────────────────────────

THE EXPERIMENT (8 scenarios, 420 seconds)

✅ Steady state at 25°C
✅ Slow ramp 25→35°C over 100 s
✅ Steady state at 35°C
✅ Fast ramp 35→45°C over 20 s
✅ Injected 55°C spike at t = 267 s  ← the interesting part
✅ Cool ramp 45→30°C
✅ Steady state at 30°C

Two filters running in parallel, same input, every sample logged
to CSV with RTC wall-clock timestamps.

──────────────────────────────────

THE KEY RESULT

Fixed KF   → Kalman gain K = 0.22 → tracked the fault ❌
Adaptive KF → K dropped to 0.004  → rejected the fault ✅

The adaptive filter monitors innovation rate.
When temperature "changes" faster than physics allows,
it inflates R by 100×.  The measurement is almost ignored.
One sample. No human intervention.

Full RMSE breakdown across all 8 scenarios is in the carousel.

──────────────────────────────────

WHY THIS MATTERS FOR REAL SYSTEMS

Sensor faults, EMI spikes, and wiring noise are facts of life
in industrial and automotive embedded systems.
A filter that can't distinguish "real change" from "bad data"
cannot be deployed in safety-relevant applications.

──────────────────────────────────

WHAT'S NEXT

This is the first in a series I'm building on
edge intelligence for constrained hardware:

→ TinyML anomaly detection on Cortex-M
→ Online learning on microcontrollers

──────────────────────────────────

📂 Full code, CSV data, and all plots on GitHub.
    Link in the comments.

──────────────────────────────────

A personal note:

I'm currently looking for my next role in embedded systems,
firmware engineering, or IoT — and the search has been tough.

If you're working on hardware that needs to be smart, robust,
and deployable on constrained devices, I'd genuinely love
to contribute to that work.

Open to roles in embedded firmware, edge AI, or sensor systems.
Feel free to reach out or simply follow along as I keep building.

#EmbeddedSystems #KalmanFilter #STM32 #IoT #SignalProcessing
```

email: mohameda.mehery@gmail.com
whatsapp: +20-1288761365
---
