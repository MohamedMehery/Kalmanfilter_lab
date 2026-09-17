#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command pipeline:  data -> train -> quantise -> evaluate -> verify edge
#
#   ./run_all.sh                  # synthetic data (IEEE C57.91 physics model)
#   ./run_all.sh mydata.csv       # your own transformer CSV
#
# Every stage is fail-fast: if the C/Python feature parity check breaks, the
# script stops rather than handing you a header that will misbehave on silicon.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
CSV="${1:-}"

hr() { printf '\n\033[1;36m%s\033[0m\n' "── $* ─────────────────────────────────────────"; }

hr "1/6  data"
if [[ -n "$CSV" ]]; then
    echo "using $CSV"
    "$PY" 01_data/preprocess.py --csv "$CSV"
else
    "$PY" 01_data/make_synthetic.py
    "$PY" 01_data/preprocess.py
fi

hr "2/6  train"
"$PY" 02_training/train.py

hr "3/6  quantise to int8 + emit C header"
"$PY" 04_deployment_esp32/scripts/convert_tflite.py

hr "4/6  evaluate"
"$PY" 03_evaluation/evaluate.py

hr "5/6  C <-> Python feature parity"
"$PY" 04_deployment_esp32/scripts/test_parity.py

hr "6/6  end-to-end edge verification"
"$PY" 04_deployment_esp32/scripts/make_replay_data.py --n 240
"$PY" 04_deployment_esp32/scripts/verify_pipeline.py

hr "done"
cat <<EOF
artifacts:
  artifacts/model_fp32.keras
  artifacts/model_int8.tflite
  04_deployment_esp32/include/transformer_pdm_model.h
  03_evaluation/report.md        <- read this
  03_evaluation/plots/*.png

flash it:
  cd 04_deployment_esp32 && pio run -t upload && pio device monitor
EOF
