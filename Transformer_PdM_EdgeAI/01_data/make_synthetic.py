"""
Physics-based synthetic transformer-monitoring data generator.
==============================================================

This is a FALLBACK so the whole pipeline can run end-to-end before the real
CSV is dropped in.  It is not a toy random-number dump: the thermal behaviour
follows the IEEE Std C57.91 clause-7 exponential (top-oil / hot-spot) model,
which is the same model utilities use for loading guides.

    Top-oil rise (steady state)
        dTheta_oil_ss = dTheta_or * ( (K^2 * R + 1) / (R + 1) ) ^ n

    Hot-spot rise (steady state)
        dTheta_hs_ss  = dTheta_hr * K ^ (2m)

    First-order lag towards steady state
        dTheta(t+dt) = dTheta(t) + (dTheta_ss - dTheta(t)) * (1 - exp(-dt/tau))

    Hot-spot temperature
        Theta_hs = Theta_ambient + dTheta_oil + dTheta_hs

where K = per-unit load, R = load-loss/no-load-loss ratio, n and m are the
oil/winding exponents (0.8 for ONAF per Table 4 of C57.91).

On top of the healthy baseline we inject realistic degradation events:
    * progressive cooling-fan / radiator fouling  -> oil rise creeps up
    * oil leak                                    -> OLI drops, cooling worsens
    * sustained overload                          -> K > 1 for hours
    * phase-voltage unbalance                     -> negative-sequence heating
    * current unbalance / partial short           -> one phase current jumps
    * sensor drift + spikes                       -> tests filter robustness

Run:
    python 01_data/make_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_CSV, SAMPLE_PERIOD_MIN, SYNTH  # noqa: E402


# ---------------------------------------------------------------- helpers
def _daily_load_shape(hours: np.ndarray) -> np.ndarray:
    """Typical distribution-transformer daily load curve (per-unit of peak)."""
    h = hours % 24.0
    morning = np.exp(-0.5 * ((h - 9.0) / 2.2) ** 2)
    evening = np.exp(-0.5 * ((h - 19.5) / 2.6) ** 2)
    base = 0.42
    return base + 0.28 * morning + 0.42 * evening


def _seasonal_ambient(day_of_year: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Ambient temperature: seasonal sine + daily sine (Cairo-ish climate)."""
    seasonal = 22.0 + 9.0 * np.sin(2 * np.pi * (day_of_year - 105.0) / 365.0)
    daily = 6.0 * np.sin(2 * np.pi * (hours % 24.0 - 9.0) / 24.0)
    return seasonal + daily


def _first_order(prev: float, target: float, dt_h: float, tau_h: float) -> float:
    """Discrete first-order lag (IEEE C57.91 exponential response)."""
    alpha = 1.0 - np.exp(-dt_h / max(tau_h, 1e-6))
    return prev + (target - prev) * alpha


# ---------------------------------------------------------------- events
def _build_events(n: int, total: int, rng: np.random.Generator) -> list[dict]:
    """Randomly place degradation events over the timeline."""
    kinds = [
        "cooling_loss", "oil_leak", "overload",
        "v_unbalance", "i_unbalance", "sensor_spike",
    ]
    # Stratified placement: one event per equal-width stratum.  Purely random
    # placement clusters events by luck, which then makes the chronological
    # train/dev/test split see very different fault rates (distribution shift
    # that has nothing to do with the physics).  Stratifying keeps the event
    # density roughly constant along the timeline.
    events: list[dict] = []
    lo, hi = 200, total - 200
    edges = np.linspace(lo, hi, n + 1).astype(int)
    for k in range(n):
        kind = kinds[k % len(kinds)] if k < len(kinds) else str(rng.choice(kinds))
        dur = int(rng.integers(8, 140))              # samples (15 min each)
        span = max(1, edges[k + 1] - edges[k] - dur)
        start = int(edges[k] + rng.integers(0, span))
        sev = float(rng.uniform(0.45, 1.0))
        events.append({"kind": kind, "start": start, "end": start + dur, "sev": sev})
    return sorted(events, key=lambda e: e["start"])


def generate(n_days: int | None = None, seed: int | None = None) -> pd.DataFrame:
    cfg = SYNTH
    n_days = int(n_days or cfg["n_days"])
    rng = np.random.default_rng(seed if seed is not None else cfg["seed"])

    dt_h = SAMPLE_PERIOD_MIN / 60.0
    n = int(n_days * 24 / dt_h)
    t_h = np.arange(n) * dt_h
    doy = (t_h / 24.0) % 365.0

    ts = pd.date_range("2024-01-01", periods=n, freq=f"{SAMPLE_PERIOD_MIN}min")

    # ---- healthy baseline drivers -------------------------------------
    load_pu = _daily_load_shape(t_h)
    load_pu *= 1.0 + 0.05 * np.sin(2 * np.pi * t_h / (24 * 7))      # weekly
    load_pu *= 1.0 + 0.12 * np.sin(2 * np.pi * (doy - 180) / 365.0)  # summer AC
    load_pu += rng.normal(0, 0.025, n)
    load_pu = np.clip(load_pu, 0.08, None)

    ambient = _seasonal_ambient(doy, t_h) + rng.normal(0, 0.6, n)

    # ---- state to be modulated by events -------------------------------
    cool_eff = np.ones(n)     # 1.0 = fans healthy, <1 = degraded cooling
    oil_level = np.full(n, 82.0) + rng.normal(0, cfg["noise"]["oli"], n)
    v_unb = np.zeros(n)       # extra voltage unbalance (%)
    i_unb = np.zeros(n)       # extra current unbalance (%)
    spike_mask = np.zeros(n, dtype=bool)

    events = _build_events(cfg["n_events"], n, rng)
    for ev in events:
        s, e, sev = ev["start"], ev["end"], ev["sev"]
        ramp = np.linspace(0.0, 1.0, e - s)
        if ev["kind"] == "cooling_loss":
            cool_eff[s:e] -= 0.45 * sev * ramp
            # Radiator fouling is cleaned at the next service visit.
            rec = min(int(24 * 4 * rng.uniform(1.0, 4.0)), n - e)
            if rec > 0:
                cool_eff[e:e + rec] -= 0.45 * sev * np.linspace(1.0, 0.0, rec)
        elif ev["kind"] == "oil_leak":
            # Oil is lost during the event, then a maintenance crew tops it up
            # over `rec` samples.  Without this recovery the degradation would
            # accumulate monotonically over the whole timeline and every later
            # sample would be CRITICAL, which is not how a maintained asset
            # behaves (and it would wreck the chronological split).
            drop = 68.0 * sev * ramp
            oil_level[s:e] -= drop
            cool_eff[s:e] -= 0.28 * sev * ramp
            rec = min(int(24 * 4 * rng.uniform(0.5, 3.0)), n - e)   # 0.5-3 days
            if rec > 0:
                heal = np.linspace(1.0, 0.0, rec)
                oil_level[e:e + rec] -= drop[-1] * heal
                cool_eff[e:e + rec] -= 0.28 * sev * heal
        elif ev["kind"] == "overload":
            bump = 0.42 * sev * np.sin(np.pi * ramp) ** 0.6
            load_pu[s:e] += bump
        elif ev["kind"] == "v_unbalance":
            v_unb[s:e] += 6.5 * sev * np.sin(np.pi * ramp) ** 0.5
        elif ev["kind"] == "i_unbalance":
            i_unb[s:e] += 34.0 * sev * np.sin(np.pi * ramp) ** 0.5
        elif ev["kind"] == "sensor_spike":
            k = max(1, int(0.12 * (e - s)))
            idx = rng.choice(np.arange(s, e), size=k, replace=False)
            spike_mask[idx] = True

    # Keep degradation inside a physically survivable band: below ~0.6 cooling
    # efficiency a real unit would trip on Buchholz / WTI protection long
    # before steady state, so simulating it adds no useful training signal.
    cool_eff = np.clip(cool_eff, 0.60, 1.05)
    oil_level = np.clip(oil_level, 8.0, 100.0)

    # ---- IEEE C57.91 thermal integration -------------------------------
    R, nn, mm = cfg["R_ratio"], cfg["n_exp"], cfg["m_exp"]
    d_or, d_hr = cfg["dtheta_or"], cfg["dtheta_hr"]
    tau_o, tau_w = cfg["tau_oil_h"], cfg["tau_wind_h"]

    d_oil = np.zeros(n)
    d_hs = np.zeros(n)
    d_oil[0] = d_or * ((load_pu[0] ** 2 * R + 1) / (R + 1)) ** nn
    d_hs[0] = d_hr * load_pu[0] ** (2 * mm)

    for i in range(1, n):
        K = load_pu[i]
        # Degraded cooling -> larger temperature rise and a slower oil time
        # constant.  Calibrated against the real ONAF -> ONAN fallback: losing
        # forced-air cooling raises the top-oil rise by roughly 30 %, it does
        # NOT scale as 1/efficiency (that would melt the unit on paper).
        deg = 1.0 / max(cool_eff[i], 0.55) - 1.0          # 0 .. 0.82
        rise_gain = 1.0 + 0.45 * deg                      # <= ~1.37
        tau_o_eff = tau_o * (1.0 + 0.8 * deg)

        oil_ss = d_or * rise_gain * ((K * K * R + 1.0) / (R + 1.0)) ** nn
        hs_ss = d_hr * rise_gain * K ** (2 * mm)
        # negative-sequence heating from unbalance (I2 losses ~ unbalance^2)
        hs_ss += 0.10 * (v_unb[i] ** 2) + 0.004 * (i_unb[i] ** 2)

        d_oil[i] = _first_order(d_oil[i - 1], oil_ss, dt_h, tau_o_eff)
        d_hs[i] = _first_order(d_hs[i - 1], hs_ss, dt_h, tau_w)

    oti = ambient + d_oil + rng.normal(0, cfg["noise"]["t"], n)
    wti = ambient + d_oil + d_hs + rng.normal(0, cfg["noise"]["t"], n)

    # ---- electrical quantities -----------------------------------------
    v_nom = cfg["v_nominal"]
    sag = 1.0 - 0.035 * np.clip(load_pu, 0, 2.0)          # voltage droop w/ load
    vl1 = v_nom * sag * (1.0 + 0.010 * v_unb / 100.0 * 3)
    vl2 = v_nom * sag * (1.0 - 0.006 * v_unb / 100.0 * 3 - v_unb / 100.0)
    vl3 = v_nom * sag * (1.0 + v_unb / 100.0 * 0.6)
    for arr in (vl1, vl2, vl3):
        arr += rng.normal(0, cfg["noise"]["v"], n)

    i_rated = cfg["i_rated"]
    ibase = load_pu * i_rated
    il1 = ibase * (1.0 + i_unb / 100.0)
    il2 = ibase * (1.0 - 0.35 * i_unb / 100.0)
    il3 = ibase * (1.0 - 0.45 * i_unb / 100.0)
    for arr in (il1, il2, il3):
        arr += rng.normal(0, cfg["noise"]["i"], n)
        np.clip(arr, 0.0, None, out=arr)

    # neutral current ~ vector sum of unbalanced phases
    inut = np.abs(il1 + il2 * np.cos(np.deg2rad(240)) + il3 * np.cos(np.deg2rad(120)))
    inut = np.abs(inut) * 0.35 + rng.normal(0, 2.0, n)
    inut = np.clip(inut, 0.0, None)

    # ---- sensor spikes (bad readings a Kalman filter must survive) ------
    sp = spike_mask
    if sp.any():
        wti[sp] += rng.normal(0, 9.0, sp.sum())
        oti[sp] += rng.normal(0, 6.0, sp.sum())
        il1[sp] *= rng.uniform(0.6, 1.5, sp.sum())

    df = pd.DataFrame({
        "timestamp": ts,
        "VL1": vl1, "VL2": vl2, "VL3": vl3,
        "IL1": il1, "IL2": il2, "IL3": il3,
        "OTI": oti, "WTI": wti, "ATI": ambient,
        "OLI": oil_level,
        "INUT": inut,
        # ground-truth extras (NOT used as model inputs — diagnostics only)
        "_load_pu": load_pu,
        "_cool_eff": cool_eff,
    })
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].round(4)
    return df


def main() -> None:
    df = generate()
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"[synthetic] wrote {len(df):,} rows -> {RAW_CSV}")
    print(f"[synthetic] span   : {df.timestamp.iloc[0]}  ->  {df.timestamp.iloc[-1]}")
    print(f"[synthetic] WTI    : {df.WTI.min():.1f} .. {df.WTI.max():.1f} degC")
    print(f"[synthetic] OTI    : {df.OTI.min():.1f} .. {df.OTI.max():.1f} degC")
    print(f"[synthetic] load pu: {df._load_pu.min():.2f} .. {df._load_pu.max():.2f}")


if __name__ == "__main__":
    main()
