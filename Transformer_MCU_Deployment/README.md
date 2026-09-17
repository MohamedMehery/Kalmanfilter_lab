# Transformer Predictive Maintenance — MCU Deployment

On-device inference firmware for the model trained in
[`../Transformer_Predictive_Maintenance/`](../Transformer_Predictive_Maintenance).
Target board: **Arduino Uno (ATmega328P)** — chosen deliberately as one of
the most resource-constrained boards commonly available (2 KB RAM, 32 KB
flash), to prove the exported model genuinely fits on a "real MCU" and not
just a beefier board. The same source builds unmodified on ESP32 / STM32
Blue Pill too (add another `[env:...]` section to `platformio.ini`).

```
Transformer_MCU_Deployment/
├── platformio.ini              # PlatformIO project config (env:uno)
├── diagram.json / wokwi.toml    # Wokwi simulation (no physical hardware needed)
├── src/
│   ├── main.cpp                 # Firmware: SELF_TEST mode + LIVE sensor mode
│   └── tfm_inference.h          # Tiny, dependency-free MLP forward-pass engine
├── model/                       # AUTO-GENERATED — copy over after retraining
│   ├── transformer_fault_model.h        # Plain-C weights + scaler (primary path)
│   ├── transformer_fault_model_tflite.h  # int8 TFLite model as C array (optional/advanced)
│   └── test_vectors.h                    # Held-out samples + reference predictions
├── test/
│   └── host_selftest.cpp        # Same self-test, compiled for your PC (no Arduino needed)
├── include/, lib/                # Standard PlatformIO folders (unused placeholders)
```

## Two deployment paths

1. **Plain-C MLP (default, recommended)** — `model/transformer_fault_model.h`
   bakes in the trained weights as `const float` arrays (stored in flash
   via `PROGMEM` on AVR) and `src/tfm_inference.h` implements the full
   `Dense→ReLU→Dense→ReLU→Dense→Sigmoid` forward pass by hand, mirroring
   the "TinyAI" style already used elsewhere in this repo (see
   `../Edge AI Driver-Behavior & Vehicle-Health Classification/`). No
   external ML library needed — total model footprint is **~2.7 KB of
   flash**, easily fitting an Arduino Uno.

2. **int8 TFLite (optional, advanced)** — `model/transformer_fault_model_tflite.h`
   is a quantized `.tflite` model exported as a C byte array, for boards
   that run [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
   (e.g. ESP32). Not wired into `main.cpp` by default since it needs the
   `tflite-micro` Arduino library, but the header is ready to use.

## Verifying the export before touching real hardware

`main.cpp` boots in **`SELF_TEST`** mode by default (`APP_MODE 0`): it
replays the bundled `test_vectors.h` samples (real held-out rows from the
training pipeline) through the on-device MLP and compares every prediction
against the reference value computed by TensorFlow on the PC/Colab side —
printing a PASS/WARNING verdict over Serial.

You can also run the exact same check **without any Arduino toolchain**,
using a plain host C++ compiler:

```bash
cd Transformer_MCU_Deployment
g++ -std=c++17 -I model -I src test/host_selftest.cpp -o /tmp/host_selftest -lm
/tmp/host_selftest
```

Expected output (numbers will vary slightly across retrains):
```
=== Transformer Fault MLP -- host self-test ===
Vectors: 20
  ...
Accuracy on bundled vectors: 17/20
Max |device_p - reference_p|: 0.000000
PASS: device inference matches PC/Colab model.
```
A `max |error|` on the order of `1e-6` or smaller confirms the C export is
numerically equivalent to the trained Keras model (any prediction
"mistakes" reported are genuine model errors, not export bugs — the
device output matches the Python reference to float precision either way).

## Building for real / simulated hardware

### Wokwi (no hardware needed)
Open this folder with the [Wokwi VS Code extension](https://docs.wokwi.com/vscode/getting-started)
(or wokwi.com) — `diagram.json` wires up two LEDs (`LED_OK` / `LED_ALERT`)
and two potentiometers standing in for the oil/ambient temperature sensors,
and `wokwi.toml` points at the PlatformIO build output.

### PlatformIO (real Arduino Uno)
```bash
pip install platformio
cd Transformer_MCU_Deployment
pio run -e uno            # build
pio run -e uno -t upload  # flash over USB
pio device monitor -b 115200
```
> Note: building requires PlatformIO to download the `atmelavr` platform
> package the first time (needs internet access to PlatformIO's registry).

## Switching to LIVE sensor mode

Set `#define APP_MODE 1` at the top of `src/main.cpp` to switch from the
bundled self-test to real analog sensor reads (pins documented at the top
of the file: `A0`=OTI, `A1`=ATI, `A2`=OLI, `A3..A5`=IL1..IL3). The firmware
builds the exact same 28-feature vector (raw readings + causal rolling
mean/std/delta) that the Python pipeline used, runs it through the tiny
MLP every 2 seconds, and drives `LED_ALERT` when the predicted fault
probability crosses 0.5.

**Before wiring real sensors**, replace the placeholder
`readVoltageLike()` scaling in `main.cpp` with your sensors' actual
calibration curves (e.g. an NTC Steinhart-Hart equation for temperature,
an ACS712 mV/A slope for current) — the demo scaling is illustrative only.

## Re-exporting after retraining

Whenever the model is retrained in
`../Transformer_Predictive_Maintenance/`, regenerate this folder's
`model/*.h` headers with:

```bash
cd ../Transformer_Predictive_Maintenance
source .venv/bin/activate
python src/export_to_mcu.py
python src/export_test_vectors.py 20
```

Both scripts write directly into `Transformer_MCU_Deployment/model/`, so
`main.cpp` and `tfm_inference.h` never need to change — only the
generated weight headers do.
