"""
Host-side parity test:  C feature engineering  ==  Python feature engineering
=============================================================================

Compiles include/pdm_features.h into a tiny host program, streams the raw test
rows through it, and diffs the resulting feature matrix against the one the
Python pipeline produced.

This is the single most valuable test in the whole project.  Train/serve skew
in the feature code is silent: the firmware still runs, the model still emits
confident probabilities, and they are simply wrong.

Run:
    python 04_deployment_esp32/scripts/test_parity.py
Exit code 0 = parity holds.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import SPLIT_DIR, SYNTH  # noqa: E402

HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>          /* atof — without this it implicitly returns int */
#include "pdm_features.h"

int main(int argc, char **argv) {
    (void)argc;
    float i_rated = (float)atof(argv[1]);
    pdm_ctx_t ctx;
    pdm_init(&ctx, i_rated);

    pdm_sample_t s;
    while (scanf("%f %f %f %f %f %f %f %f %f %f",
                 &s.vl1, &s.vl2, &s.vl3, &s.il1, &s.il2, &s.il3,
                 &s.oti, &s.wti, &s.ati, &s.oli) == 10) {
        int ready = pdm_push(&ctx, &s);
        /* always print the newest feature row so every sample is comparable */
        const float *f = ctx.feat[PDMF_WINDOW - 1];
        printf("%d", ready);
        for (int i = 0; i < PDMF_N_FEATURES; i++) printf(" %.6f", f[i]);
        printf("\n");
    }
    return 0;
}
"""


def main() -> int:
    inc = ROOT / "04_deployment_esp32" / "include"
    df = pd.read_csv(SPLIT_DIR / "processed_full.csv")
    meta = json.loads((SPLIT_DIR / "meta.json").read_text())
    feats = meta["features"]
    i_rated = float(meta["i_rated"]) or SYNTH["i_rated"]

    n = min(4000, len(df))
    sub = df.iloc[:n]

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "harness.c"
        src.write_text(HARNESS)
        exe = Path(td) / "harness"
        cc = subprocess.run(
            ["gcc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
             f"-I{inc}", str(src), "-o", str(exe), "-lm"],
            capture_output=True, text=True)
        if cc.returncode != 0:
            print("[parity] COMPILE FAILED\n" + cc.stderr)
            return 1
        print("[parity] C harness compiled OK")

        cols = ["VL1", "VL2", "VL3", "IL1", "IL2", "IL3", "OTI", "WTI", "ATI", "OLI"]
        stdin = "\n".join(
            " ".join(f"{v:.6f}" for v in row) for row in sub[cols].to_numpy(float))
        run = subprocess.run([str(exe), f"{i_rated:.6f}"], input=stdin,
                             capture_output=True, text=True)
        if run.returncode != 0:
            print("[parity] RUN FAILED\n" + run.stderr)
            return 1

    out = np.array([[float(x) for x in line.split()]
                    for line in run.stdout.strip().splitlines()])
    c_feats = out[:, 1:]
    py_feats = sub[feats].to_numpy(float)

    assert c_feats.shape == py_feats.shape, (c_feats.shape, py_feats.shape)

    diff = np.abs(c_feats - py_feats)
    # ignore the first LAG_1H rows: pandas .diff(4) has no history there and
    # emits 0.0 after fillna, which the C code reproduces only once its tail
    # buffer is primed.  Both are "warm-up" rows and are never fed to the NN.
    warm = 4
    d = diff[warm:]

    print(f"[parity] compared {len(d):,} rows x {len(feats)} features")
    worst = d.max(axis=0)
    ok = True
    for i, f in enumerate(feats):
        tol = 2e-3 if f in ("I_avg", "load_pu", "IL1", "IL2", "IL3") else 5e-4
        flag = "ok " if worst[i] <= tol else "FAIL"
        if worst[i] > tol:
            ok = False
        if worst[i] > 1e-6 or not ok:
            print(f"  {flag} {f:14s} max|diff| = {worst[i]:.3e}  (tol {tol:.0e})")
    print(f"[parity] overall max diff = {d.max():.3e}")

    if ok:
        print("[parity] PASS — C and Python feature pipelines agree")
        return 0
    print("[parity] FAIL — fix pdm_features.h before flashing")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
