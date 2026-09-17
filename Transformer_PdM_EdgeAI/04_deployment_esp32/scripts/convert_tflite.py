"""
Convert the trained Keras model to full-integer int8 TFLite + a C array.
========================================================================

Why full-integer int8
---------------------
* ~4x smaller than float32 and the ESP32 has no FPU-friendly SIMD for NN work,
  so int8 kernels are also materially faster.
* TFLite-Micro's reference kernels for int8 are the best-tested path.

The input/output tensors are kept int8 as well (not just the weights), which
removes the float<->int8 conversion ops at the graph boundary.  The firmware
therefore has to quantise the normalised window itself -- the scale and
zero-point for that are emitted into the header.

Numerical verification is NOT optional here: the script re-runs the whole test
split through the interpreter and reports the agreement with the float model.
A quantised model that silently loses the rare CRITICAL class is a classic
TinyML failure mode.

Run:
    python 04_deployment_esp32/scripts/convert_tflite.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ARTIFACT_DIR, CLASS_NAMES, DEPLOY_DIR, EDGE, SPLIT_DIR  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402


def representative_dataset(X: np.ndarray, n: int):
    """Calibration set: a stratified slice of the TRAIN split."""
    idx = np.linspace(0, len(X) - 1, min(n, len(X))).astype(int)
    for i in idx:
        yield [X[i: i + 1].astype(np.float32)]


def convert(model, X_calib: np.ndarray) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: representative_dataset(
        X_calib, EDGE["rep_samples"])
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def tflite_infer(tflite_bytes: bytes, X: np.ndarray, batch_log: int = 2000):
    """Run the int8 interpreter over X, returning (probs, wti_std_units)."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()

    in_scale, in_zp = inp["quantization"]
    probs = np.zeros((len(X), len(CLASS_NAMES)), np.float32)
    regs = np.zeros(len(X), np.float32)

    # identify which output is which by shape (3 -> health, 1 -> wti)
    o_cls = next(o for o in outs if o["shape"][-1] == len(CLASS_NAMES))
    o_reg = next(o for o in outs if o["shape"][-1] == 1)

    for i in range(len(X)):
        q = np.round(X[i] / in_scale + in_zp).astype(np.int8)
        interp.set_tensor(inp["index"], q[None, ...])
        interp.invoke()
        cs, czp = o_cls["quantization"]
        rs, rzp = o_reg["quantization"]
        probs[i] = (interp.get_tensor(o_cls["index"])[0].astype(np.float32) - czp) * cs
        regs[i] = (interp.get_tensor(o_reg["index"])[0][0].astype(np.float32) - rzp) * rs
        if batch_log and i and i % batch_log == 0:
            print(f"    ... {i}/{len(X)}")
    return probs, regs, (float(in_scale), int(in_zp))


def emit_header(tflite_bytes: bytes, norm, op, meta, path: Path) -> None:
    mean, std, r_mean, r_std = norm
    in_scale, in_zp = op["input_quant"]
    b = tflite_bytes
    lines = []
    ap = lines.append
    ap("/*")
    ap(" * AUTO-GENERATED -- do not edit by hand.")
    ap(" * Source : 04_deployment_esp32/scripts/convert_tflite.py")
    ap(" *")
    ap(" * Transformer predictive-maintenance model (int8, TFLite-Micro)")
    ap(f" *   window   : {meta['window']} samples x {meta['n_features']} features")
    ap(f" *   horizon  : {meta['horizon']} samples ahead")
    ap(f" *   size     : {len(b)} bytes")
    ap(" */")
    ap("#ifndef TRANSFORMER_PDM_MODEL_H")
    ap("#define TRANSFORMER_PDM_MODEL_H")
    ap("")
    ap("#include <stdint.h>")
    ap("")
    ap(f"#define PDM_WINDOW        {meta['window']}")
    ap(f"#define PDM_N_FEATURES    {meta['n_features']}")
    ap(f"#define PDM_HORIZON       {meta['horizon']}")
    ap(f"#define PDM_N_CLASSES     {len(CLASS_NAMES)}")
    ap(f"#define PDM_TENSOR_ARENA  ({EDGE['tensor_arena_kb']} * 1024)")
    ap("")
    ap("/* input quantisation:  q = round(x / scale) + zero_point  */")
    ap(f"#define PDM_IN_SCALE      {in_scale:.8e}f")
    ap(f"#define PDM_IN_ZERO_PT    {in_zp}")
    ap("")
    ap("/* alarm fires when P(WARNING)+P(CRITICAL) >= threshold  */")
    ap(f"#define PDM_ALARM_THRESH  {op['alarm_threshold']:.5f}f")
    ap("")
    ap("/* WTI regression head is standardised: degC = y*r_std + r_mean */")
    ap(f"#define PDM_WTI_MEAN      {float(r_mean):.6f}f")
    ap(f"#define PDM_WTI_STD       {float(r_std):.6f}f")
    ap("")
    ap("/* per-feature normalisation (must match training exactly) */")
    ap(f"static const float pdm_norm_mean[PDM_N_FEATURES] = {{")
    ap("  " + ", ".join(f"{v:.6f}f" for v in mean))
    ap("};")
    ap(f"static const float pdm_norm_scale[PDM_N_FEATURES] = {{")
    ap("  " + ", ".join(f"{v:.6f}f" for v in std))
    ap("};")
    ap("")
    ap("static const char* const pdm_class_name[PDM_N_CLASSES] = {"
       + ", ".join(f'"{c}"' for c in CLASS_NAMES) + "};")
    ap("")
    ap("/* feature order expected in every window row */")
    ap("/* " + ", ".join(meta["features"]) + " */")
    ap("")
    ap(f"const unsigned int transformer_pdm_model_len = {len(b)};")
    ap("alignas(8) const unsigned char transformer_pdm_model[] = {")
    for i in range(0, len(b), 12):
        ap("  " + ", ".join(f"0x{x:02x}" for x in b[i:i + 12]) + ",")
    ap("};")
    ap("")
    ap("#endif  /* TRANSFORMER_PDM_MODEL_H */")
    path.write_text("\n".join(lines))


def main() -> None:
    meta = json.loads((SPLIT_DIR / "meta.json").read_text())
    op = json.loads((ARTIFACT_DIR / "operating_point.json").read_text())
    nz = np.load(ARTIFACT_DIR / "norm.npz")
    mean, std = nz["mean"], nz["std"]
    r_mean, r_std = float(nz["r_mean"]), float(nz["r_std"])

    model = keras.models.load_model(ARTIFACT_DIR / "model_fp32.keras", compile=False)

    d = np.load(SPLIT_DIR / "dataset.npz")
    Xtr = ((d["X_train"] - mean) / std).astype(np.float32)
    Xte = ((d["X_test"] - mean) / std).astype(np.float32)
    yte_c, yte_r = d["y_cls_test"], d["y_reg_test"]

    print("[convert] converting to int8 TFLite ...")
    blob = convert(model, Xtr)
    out_tflite = ARTIFACT_DIR / "model_int8.tflite"
    out_tflite.write_bytes(blob)
    print(f"[convert] tflite size   : {len(blob):,} bytes "
          f"({len(blob)/1024:.1f} KB)")

    # ---------------------------------------------- numerical verification
    print("[convert] verifying int8 vs float32 on the TEST split ...")
    fp = model.predict(Xte, verbose=0)
    fp_prob, fp_reg = fp[0], fp[1].ravel()
    q_prob, q_reg, in_q = tflite_infer(blob, Xte)

    agree = float((fp_prob.argmax(1) == q_prob.argmax(1)).mean())
    risk_f = fp_prob[:, 1] + fp_prob[:, 2]
    risk_q = q_prob[:, 1] + q_prob[:, 2]
    thr = op["alarm_threshold"]
    alarm_agree = float(((risk_f >= thr) == (risk_q >= thr)).mean())

    wti_f = fp_reg * r_std + r_mean
    wti_q = q_reg * r_std + r_mean
    mae_f = float(np.abs(wti_f - yte_r).mean())
    mae_q = float(np.abs(wti_q - yte_r).mean())

    a = (yte_c > 0).astype(int)
    p = (risk_q >= thr).astype(int)
    tp = int(((p == 1) & (a == 1)).sum()); fp_ = int(((p == 1) & (a == 0)).sum())
    fn = int(((p == 0) & (a == 1)).sum()); tn = int(((p == 0) & (a == 0)).sum())
    prec = tp / (tp + fp_) if tp + fp_ else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    print(f"[convert] argmax agreement  : {agree*100:6.2f} %")
    print(f"[convert] alarm-bit agree   : {alarm_agree*100:6.2f} %")
    print(f"[convert] WTI MAE  float    : {mae_f:6.3f} degC")
    print(f"[convert] WTI MAE  int8     : {mae_q:6.3f} degC  "
          f"(delta {mae_q-mae_f:+.3f})")
    print(f"[convert] int8 alarm  prec {prec*100:5.1f} %  rec {rec*100:5.1f} %  "
          f"FAR {fp_/(fp_+tn)*100:4.2f} %")

    # ---------------------------------------------------------- C header
    hdr = DEPLOY_DIR / "include" / "transformer_pdm_model.h"
    hdr.parent.mkdir(parents=True, exist_ok=True)
    emit_header(blob, (mean, std, r_mean, r_std),
                {**op, "input_quant": in_q}, meta, hdr)
    print(f"[convert] header written    : {hdr}  ({hdr.stat().st_size/1024:.1f} KB)")

    (ARTIFACT_DIR / "quantization_report.json").write_text(json.dumps({
        "tflite_bytes": len(blob),
        "argmax_agreement": agree,
        "alarm_agreement": alarm_agree,
        "wti_mae_float": mae_f, "wti_mae_int8": mae_q,
        "int8_alarm": {"precision": prec, "recall": rec,
                       "tp": tp, "fp": fp_, "fn": fn, "tn": tn},
        "input_quant": {"scale": in_q[0], "zero_point": in_q[1]},
    }, indent=2))


if __name__ == "__main__":
    main()
