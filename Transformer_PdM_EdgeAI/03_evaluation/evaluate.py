"""
Full evaluation: float vs int8, against honest baselines, with plots.
=====================================================================

Baselines every claim is measured against
-----------------------------------------
* PERSISTENCE  : "the state in H steps == the state now".  For a thermally
                 slow asset this is very strong and must be reported.
* RULE-ON-NOW  : the IEEE/NEMA threshold rules applied to the current sample.
* PREDICTED-RULE: apply the same thresholds to the model's forecast WTI.

Outputs -> 03_evaluation/plots/*.png and 03_evaluation/report.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ARTIFACT_DIR, CLASS_NAMES, EVAL_DIR, SPLIT_DIR, THRESH  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

PLOTS = EVAL_DIR / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------- metrics
def confusion(y, p, k=3):
    m = np.zeros((k, k), int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def prf(y, p, k=3):
    rows = []
    for c in range(k):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        rows.append((CLASS_NAMES[c], pr, rc, f1, int((y == c).sum())))
    macro = float(np.mean([r[3] for r in rows]))
    return rows, macro


def alarm_stats(y_cls, risk, thr):
    a = (y_cls > 0).astype(int)
    p = (risk >= thr).astype(int)
    tp = int(((p == 1) & (a == 1)).sum()); fp = int(((p == 1) & (a == 0)).sum())
    fn = int(((p == 0) & (a == 1)).sum()); tn = int(((p == 0) & (a == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": pr, "recall": rc,
        "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
        "far": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def int8_predict(tflite_path: Path, X: np.ndarray):
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    o_cls = next(o for o in outs if o["shape"][-1] == len(CLASS_NAMES))
    o_reg = next(o for o in outs if o["shape"][-1] == 1)
    s, z = inp["quantization"]
    P = np.zeros((len(X), len(CLASS_NAMES)), np.float32)
    R = np.zeros(len(X), np.float32)
    for i in range(len(X)):
        interp.set_tensor(inp["index"],
                          np.round(X[i] / s + z).astype(np.int8)[None, ...])
        interp.invoke()
        cs, cz = o_cls["quantization"]
        rs, rz = o_reg["quantization"]
        P[i] = (interp.get_tensor(o_cls["index"])[0].astype(np.float32) - cz) * cs
        R[i] = (interp.get_tensor(o_reg["index"])[0][0].astype(np.float32) - rz) * rs
    return P, R


# ---------------------------------------------------------------- main
def main() -> None:
    meta = json.loads((SPLIT_DIR / "meta.json").read_text())
    op = json.loads((ARTIFACT_DIR / "operating_point.json").read_text())
    nz = np.load(ARTIFACT_DIR / "norm.npz")
    mean, std = nz["mean"], nz["std"]
    r_mean, r_std = float(nz["r_mean"]), float(nz["r_std"])
    thr = op["alarm_threshold"]

    d = np.load(SPLIT_DIR / "dataset.npz")
    Xte = ((d["X_test"] - mean) / std).astype(np.float32)
    yc, yr, ynow = d["y_cls_test"], d["y_reg_test"], d["y_now_test"]

    model = keras.models.load_model(ARTIFACT_DIR / "model_fp32.keras", compile=False)
    fprob, freg = model.predict(Xte, verbose=0)
    freg = freg.ravel() * r_std + r_mean
    fpred = fprob.argmax(1)
    frisk = fprob[:, 1] + fprob[:, 2]

    tfl = ARTIFACT_DIR / "model_int8.tflite"
    qprob, qreg = int8_predict(tfl, Xte)
    qreg = qreg * r_std + r_mean
    qpred = qprob.argmax(1)
    qrisk = qprob[:, 1] + qprob[:, 2]

    # ---- baselines ----------------------------------------------------
    pers = ynow
    li = {f: i for i, f in enumerate(meta["features"])}
    last = d["X_test"][:, -1, :]

    def rule_sev(WTI, OTI, OLI, Vu, Iu, L):
        sev = np.zeros(len(WTI), int)

        def bump(w, c):
            nonlocal sev
            sev = np.maximum(sev, np.where(c, 2, np.where(w, 1, 0)))
        bump(WTI >= THRESH["wti_warn"], WTI >= THRESH["wti_crit"])
        bump(OTI >= THRESH["oti_warn"], OTI >= THRESH["oti_crit"])
        bump(OLI <= THRESH["oli_warn"], OLI <= THRESH["oli_crit"])
        bump(Vu >= THRESH["unbal_warn"], Vu >= THRESH["unbal_crit"])
        bump(Iu >= THRESH["iunbal_warn"], Iu >= THRESH["iunbal_crit"])
        bump(L >= THRESH["load_warn"], L >= THRESH["load_crit"])
        return sev

    rule_now = rule_sev(last[:, li["WTI"]], last[:, li["OTI"]], last[:, li["OLI"]],
                        last[:, li["V_unbal"]], last[:, li["I_unbal"]],
                        last[:, li["load_pu"]])
    # rule applied to the FORECAST winding temperature (others held at now)
    rule_fc = rule_sev(freg, last[:, li["OTI"]], last[:, li["OLI"]],
                       last[:, li["V_unbal"]], last[:, li["I_unbal"]],
                       last[:, li["load_pu"]])

    rows = []
    for name, pred in (("persistence", pers), ("rule-on-now", rule_now),
                       ("rule-on-forecast", rule_fc),
                       ("CNN float32", fpred), ("CNN int8", qpred)):
        _, macro = prf(yc, pred)
        rows.append((name, float((pred == yc).mean()), macro))

    ew = (ynow == 0) & (yc > 0)
    ew_f = float((fpred[ew] > 0).mean()) if ew.any() else float("nan")
    ew_q = float((qpred[ew] > 0).mean()) if ew.any() else float("nan")
    ew_rule = float((rule_fc[ew] > 0).mean()) if ew.any() else float("nan")

    af = alarm_stats(yc, frisk, thr)
    aq = alarm_stats(yc, qrisk, thr)

    mae_f = float(np.abs(freg - yr).mean())
    mae_q = float(np.abs(qreg - yr).mean())
    rmse_f = float(np.sqrt(((freg - yr) ** 2).mean()))
    # naive forecast: WTI stays where it is now
    naive = d["X_test"][:, -1, li["WTI"]]
    mae_naive = float(np.abs(naive - yr).mean())

    # ------------------------------------------------------------ plots
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    # 1 confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (nm, pred) in zip(axes, (("Persistence", pers),
                                     ("CNN float32", fpred), ("CNN int8", qpred))):
        cm = confusion(yc, pred)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES, rotation=30, fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES, fontsize=7)
        ax.set_title(nm, fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.suptitle("Confusion matrices — test split (future state)", fontsize=10)
    fig.tight_layout(); fig.savefig(PLOTS / "01_confusion.png"); plt.close(fig)

    # 2 WTI forecast timeline
    fig, ax = plt.subplots(figsize=(11, 3.2))
    n = min(900, len(yr))
    ax.plot(yr[:n], lw=1.1, label="true WTI @ t+H", color="#222")
    ax.plot(freg[:n], lw=1.0, label=f"float32 (MAE {mae_f:.2f} °C)", color="#1f77b4")
    ax.plot(qreg[:n], lw=0.9, ls="--", label=f"int8 (MAE {mae_q:.2f} °C)", color="#d62728")
    ax.axhline(THRESH["wti_warn"], color="orange", ls=":", lw=1, label="warning 95 °C")
    ax.set_xlabel("test sample"); ax.set_ylabel("winding temp (°C)")
    ax.set_title("Winding-temperature forecast, 2 h ahead"); ax.legend(fontsize=7, ncol=4)
    fig.tight_layout(); fig.savefig(PLOTS / "02_wti_forecast.png"); plt.close(fig)

    # 3 risk trace + alarms
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.plot(qrisk[:n], lw=0.9, color="#8c564b", label="risk = P(WARN)+P(CRIT)")
    ax.axhline(thr, color="red", ls="--", lw=1, label=f"threshold {thr:.2f}")
    ax.fill_between(range(n), 0, 1, where=(yc[:n] > 0), color="orange",
                    alpha=0.25, step="mid", label="true degraded")
    ax.set_ylim(0, 1.02); ax.set_xlabel("test sample"); ax.set_ylabel("risk")
    ax.set_title("Alarm risk signal (int8 model)"); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(PLOTS / "03_risk_trace.png"); plt.close(fig)

    # 4 precision/recall vs threshold
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ts = np.linspace(0.02, 0.99, 120)
    P = [alarm_stats(yc, qrisk, t)["precision"] for t in ts]
    R = [alarm_stats(yc, qrisk, t)["recall"] for t in ts]
    F = [alarm_stats(yc, qrisk, t)["f1"] for t in ts]
    ax.plot(ts, P, label="precision"); ax.plot(ts, R, label="recall")
    ax.plot(ts, F, label="F1", ls="--")
    ax.axvline(thr, color="red", lw=1, ls=":", label="chosen")
    ax.set_xlabel("alarm threshold"); ax.set_ylabel("score")
    ax.set_title("Alarm operating curve (int8)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(PLOTS / "04_alarm_curve.png"); plt.close(fig)

    # 5 training history
    hp = ARTIFACT_DIR / "train_history.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
        axs[0].plot(h.get("loss", []), label="train"); axs[0].plot(h.get("val_loss", []), label="dev")
        axs[0].set_title("total loss"); axs[0].legend(fontsize=7); axs[0].set_xlabel("epoch")
        axs[1].plot(h.get("wti_mae", []), label="train"); axs[1].plot(h.get("val_wti_mae", []), label="dev")
        axs[1].set_title("WTI MAE (standardised)"); axs[1].legend(fontsize=7); axs[1].set_xlabel("epoch")
        axs[2].plot(h.get("val_alarm_f1", []), color="green")
        axs[2].set_title("dev alarm-F1 (model selection)"); axs[2].set_xlabel("epoch")
        fig.tight_layout(); fig.savefig(PLOTS / "05_training.png"); plt.close(fig)

    # 6 float vs int8 scatter
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(freg, qreg, s=3, alpha=0.3, color="#2ca02c")
    lim = [min(freg.min(), qreg.min()), max(freg.max(), qreg.max())]
    ax.plot(lim, lim, color="black", lw=0.8, ls="--")
    ax.set_xlabel("float32 forecast (°C)"); ax.set_ylabel("int8 forecast (°C)")
    ax.set_title("Quantisation fidelity")
    fig.tight_layout(); fig.savefig(PLOTS / "06_quant_fidelity.png"); plt.close(fig)

    # ----------------------------------------------------------- report
    qr = json.loads((ARTIFACT_DIR / "quantization_report.json").read_text())
    L = []
    L.append("# Evaluation report — Transformer predictive maintenance\n")
    L.append(f"Test windows: **{len(yc):,}**  |  horizon: **{meta['horizon']} samples "
             f"({meta['horizon']*15/60:.1f} h)**  |  window: **{meta['window']} samples**\n")
    L.append("\n## 1. Three-class future-state accuracy\n")
    L.append("| model | accuracy | macro-F1 |")
    L.append("|---|---|---|")
    for nm, a, m in rows:
        L.append(f"| {nm} | {a*100:.2f} % | {m:.4f} |")
    L.append("\n> The 3-class task is **persistence-dominated**: the state rarely "
             "changes inside one horizon, so a model that simply echoes the current "
             "state already scores very high accuracy. Accuracy alone is therefore "
             "*not* evidence that the model learned anything. The metrics that matter "
             "are below.\n")

    L.append("\n## 2. Early warning (the part that is actually useful)\n")
    L.append(f"Windows that look healthy **now** but degrade within the horizon: "
             f"**{int(ew.sum())}**\n")
    L.append("| model | early-warning recall |")
    L.append("|---|---|")
    L.append(f"| persistence | 0.0 % (0 by construction) |")
    L.append(f"| rule on forecast WTI | {ew_rule*100:.1f} % |")
    L.append(f"| CNN float32 | {ew_f*100:.1f} % |")
    L.append(f"| CNN int8 | {ew_q*100:.1f} % |")

    L.append("\n## 3. Alarm head (deployed decision)\n")
    L.append(f"`risk = P(WARNING) + P(CRITICAL)`, threshold **{thr:.3f}** "
             f"chosen on the dev split at a {op.get('far_budget', 0.01)*100:.1f} % "
             "false-alarm budget.\n")
    L.append("| model | precision | recall | F1 | false-alarm rate |")
    L.append("|---|---|---|---|---|")
    L.append(f"| float32 | {af['precision']*100:.1f} % | {af['recall']*100:.1f} % "
             f"| {af['f1']:.3f} | {af['far']*100:.2f} % |")
    L.append(f"| int8 | {aq['precision']*100:.1f} % | {aq['recall']*100:.1f} % "
             f"| {aq['f1']:.3f} | {aq['far']*100:.2f} % |")

    L.append("\n## 4. Winding-temperature forecast\n")
    L.append("| model | MAE (°C) | RMSE (°C) |")
    L.append("|---|---|---|")
    L.append(f"| naive (WTI stays put) | {mae_naive:.3f} | — |")
    L.append(f"| CNN float32 | {mae_f:.3f} | {rmse_f:.3f} |")
    L.append(f"| CNN int8 | {mae_q:.3f} | — |")
    L.append(f"\nTarget std on test: {yr.std():.2f} °C. The regression head beats the "
             f"naive carry-forward by {(1-mae_f/mae_naive)*100:.0f} %.\n")

    L.append("\n## 5. Quantisation\n")
    L.append(f"- TFLite size: **{qr['tflite_bytes']:,} bytes "
             f"({qr['tflite_bytes']/1024:.1f} KB)**")
    L.append(f"- argmax agreement float vs int8: **{qr['argmax_agreement']*100:.2f} %**")
    L.append(f"- alarm-bit agreement: **{qr['alarm_agreement']*100:.2f} %**")
    L.append(f"- WTI MAE change: {qr['wti_mae_float']:.3f} -> "
             f"{qr['wti_mae_int8']:.3f} °C\n")
    L.append("\n## Plots\n")
    for p in sorted(PLOTS.glob("*.png")):
        L.append(f"![{p.stem}](plots/{p.name})")
    (EVAL_DIR / "report.md").write_text("\n".join(L))

    print("\n".join(L[:40]))
    print(f"\n[eval] report -> {EVAL_DIR/'report.md'}")
    print(f"[eval] plots  -> {PLOTS}")


if __name__ == "__main__":
    main()
