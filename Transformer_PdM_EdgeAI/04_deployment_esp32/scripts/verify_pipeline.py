"""
End-to-end edge verification, no hardware required.
===================================================

    C feature pipeline (sim_main.cpp)  ->  int8 TFLite interpreter
                                       vs
    host reference decisions baked into replay_data.h

If this passes, the only things left that can differ on the real ESP32 are the
sensor front-end and the TFLM kernel implementations — the data path, the
normalisation, the quantisation and the decision logic are all proven.

Run:
    python 04_deployment_esp32/scripts/verify_pipeline.py
Exit 0 = the edge path reproduces the host results.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import ARTIFACT_DIR, DEPLOY_DIR, N_CLASSES, WINDOW  # noqa: E402

import tensorflow as tf  # noqa: E402


def parse_expected(hdr_text: str):
    body = hdr_text.split("replay_expected[REPLAY_EXPECTED_N] = {")[1].split("};")[0]
    rows = re.findall(r"\{(\d+),\s*([-\d.eE+]+)f,\s*(\d+),\s*([-\d.eE+]+)f\}", body)
    return [(int(a), float(b), int(c), float(d)) for a, b, c, d in rows]


def main() -> int:
    inc = DEPLOY_DIR / "include"
    src = DEPLOY_DIR / "src" / "sim_main.cpp"
    hdr = (inc / "replay_data.h").read_text()
    expected = parse_expected(hdr)
    thr = float(re.search(r"REPLAY_ALARM_THRESH\s+([-\d.eE+]+)f", hdr).group(1))

    # ---- build + run the C simulator ----------------------------------
    with tempfile.TemporaryDirectory() as td:
        exe = Path(td) / "sim"
        cc = subprocess.run(
            ["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror",
             f"-I{inc}", str(src), "-o", str(exe)],
            capture_output=True, text=True)
        if cc.returncode != 0:
            print("[verify] COMPILE FAILED\n" + cc.stderr)
            return 1
        print("[verify] native simulator compiled OK")
        run = subprocess.run([str(exe), "1449.0"], capture_output=True, text=True)
        if run.returncode != 0:
            print("[verify] RUN FAILED\n" + run.stderr)
            return 1
    print("[verify] " + run.stderr.strip())

    lines = run.stdout.strip().splitlines()
    rows, windows = [], []
    for ln in lines:
        p = ln.split()
        rows.append(int(p[0]))
        windows.append(np.array(p[1:], np.float32).reshape(WINDOW, -1))
    windows = np.stack(windows)
    print(f"[verify] windows from C: {windows.shape}")

    # ---- run them through the int8 interpreter ------------------------
    nz = np.load(ARTIFACT_DIR / "norm.npz")
    mean, std = nz["mean"], nz["std"]
    r_mean, r_std = float(nz["r_mean"]), float(nz["r_std"])

    interp = tf.lite.Interpreter(model_path=str(ARTIFACT_DIR / "model_int8.tflite"))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    o_cls = next(o for o in outs if o["shape"][-1] == N_CLASSES)
    o_reg = next(o for o in outs if o["shape"][-1] == 1)
    s_in, z_in = inp["quantization"]

    got = []
    for w in windows:
        x = (w - mean) / std
        q = np.clip(np.round(x / s_in) + z_in, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None, ...])
        interp.invoke()
        cs, cz = o_cls["quantization"]
        rs, rz = o_reg["quantization"]
        p = (interp.get_tensor(o_cls["index"])[0].astype(np.float32) - cz) * cs
        p = p / max(p.sum(), 1e-6)
        r = (interp.get_tensor(o_reg["index"])[0][0].astype(np.float32) - rz) * rs
        got.append((float(p[1] + p[2]), int(p.argmax()), float(r * r_std + r_mean)))

    # ---- compare ------------------------------------------------------
    exp_map = {e[0]: e for e in expected}
    n_cmp = 0
    d_risk, d_wti, cls_mismatch, alarm_mismatch = [], [], 0, 0

    # Cold-start skew, and why it is excluded
    # ---------------------------------------
    # The Python reference builds its features from the FULL processed CSV, so
    # at replay row 15 it already knows OTI/WTI from four samples earlier (the
    # 1-hour trend).  A freshly booted MCU does not: its ring buffer is empty,
    # so WTI_trend_1h / OTI_trend_1h read 0 until PDMF_LAG_1H samples have been
    # pushed.  The first WINDOW+LAG_1H-1 rows are therefore expected to differ
    # and the firmware must not act on them.  This is a real deployment
    # property, not a bug: see PDM_WARMUP_SAMPLES in pdm_features.h.
    warmup_until = min(rows) + 4 if rows else 0
    skipped = 0

    for row, (risk, cls, wfc) in zip(rows, got):
        if row not in exp_map:
            continue
        if row < warmup_until:
            skipped += 1
            continue
        _, e_risk, e_cls, e_wfc = exp_map[row]
        n_cmp += 1
        d_risk.append(abs(risk - e_risk))
        d_wti.append(abs(wfc - e_wfc))
        if cls != e_cls:
            cls_mismatch += 1
        if (risk >= thr) != (e_risk >= thr):
            alarm_mismatch += 1

    if not n_cmp:
        print("[verify] FAIL — nothing compared (row alignment is off)")
        return 1

    d_risk = np.array(d_risk); d_wti = np.array(d_wti)
    print(f"[verify] warm-up rows skipped : {skipped} "
          f"(MCU has no 1-h history before row {warmup_until})")
    print(f"[verify] compared      : {n_cmp} inferences")
    print(f"[verify] max |d risk|  : {d_risk.max():.6f}")
    print(f"[verify] max |d WTI|   : {d_wti.max():.4f} degC")
    print(f"[verify] class mismatch: {cls_mismatch}")
    print(f"[verify] alarm mismatch: {alarm_mismatch}")

    ok = (d_risk.max() < 1e-3 and d_wti.max() < 1e-2
          and cls_mismatch == 0 and alarm_mismatch == 0)
    if ok:
        print("[verify] PASS — edge path reproduces the host decisions exactly")
        return 0
    print("[verify] FAIL — investigate before flashing")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
