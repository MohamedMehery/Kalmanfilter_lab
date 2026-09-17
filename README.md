# Kalmanfilter_lab
Kalman filter on embedded devices

## DHT22_temprature_humidty

This folder contains an STM32 Bluepill Arduino project that reads temperature and humidity from a DHT22 sensor, displays values on an I2C LCD2004, and optionally uses a DS1307 RTC module. The code is designed for simulation and embedded development using PlatformIO and Wokwi.

### main.ino functionality

- Implements a custom bit-banged I2C driver for the LCD and RTC on pins `PB6`/`PB7`.
- Reads DHT22 sensor data from pin `PA0` with a dedicated `dht22_read()` driver.
- Applies a Kalman filter to temperature readings to smooth noisy measurements:
  - initializes the filter from the first reliable DHT22 reading
  - uses process noise `Q = 0.01` and measurement noise `R = 0.5`
  - updates filtered temperature every 2 seconds
- Updates the LCD display every 500ms with raw temperature, filtered temperature, and humidity.
- Includes debug logging and optional feature toggles for LCD, RTC, DHT, and Kalman filter behavior.

### Build and simulation

- Use `platformio.ini` to build the STM32 Bluepill firmware.
- Use `wokwi.toml` and `wokwi_schematic.json` to run the simulation in Wokwi.

## Transformer_PdM_EdgeAI

Predictive-maintenance TinyML pipeline for power transformers, from dataset to
verified ESP32 firmware.

- **Model**: multi-head 1-D CNN (8,204 params) predicting, 2 hours ahead, the
  health state (NORMAL/WARNING/CRITICAL), an alarm bit, and the winding
  temperature in degC.
- **Pipeline**: `./run_all.sh` runs data -> train/dev/test -> int8 quantisation
  -> evaluation -> edge verification in about 4 minutes. A Colab notebook is in
  `05_notebooks/`.
- **Deployment**: 17.5 KB int8 TFLite embedded as a C array, 24 KB tensor
  arena, ~4 ms per inference on an ESP32. PlatformIO + Wokwi project included.
- **Honesty about results**: the 3-class task is persistence-dominated (a
  "nothing changes" baseline scores 95.5 %), so the project is evaluated on
  what persistence cannot do: 33 % early-warning recall on cases that look
  healthy now but degrade within 2 h, and a WTI forecast MAE of 2.66 degC
  versus 6.25 degC for naive carry-forward.
- **Verification**: two gates run before any flash - a C-vs-Python feature
  parity test (agreement to 1.5e-4) and a full edge-path check proving the
  firmware reproduces host decisions exactly (0 class or alarm mismatches).

See [`Transformer_PdM_EdgeAI/README.md`](Transformer_PdM_EdgeAI/README.md).
