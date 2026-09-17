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

## Transformer_Predictive_Maintenance + Transformer_MCU_Deployment

A two-stage predictive-maintenance project for a real distribution
transformer, built on the [Distributed Transformer Monitoring](https://www.kaggle.com/datasets/sreshta140/ai-transformer-monitoring)
Kaggle dataset (IoT sensor readings every ~15 minutes, 2019-06-25 to
2020-04-14):

- **[`Transformer_Predictive_Maintenance/`](Transformer_Predictive_Maintenance)** —
  Python data pipeline + classical ML / deep learning training, runnable
  locally or on Google Colab. Loads the raw sensor CSVs, engineers causal
  rolling-window features, builds a stratified train/dev/test split, trains
  and compares Logistic Regression / Random Forest / Gradient Boosting
  baselines against a small MCU-friendly Keras MLP, and exports everything
  needed for on-device deployment.
- **[`Transformer_MCU_Deployment/`](Transformer_MCU_Deployment)** —
  Arduino/PlatformIO firmware (Uno-ready, ~2.7 KB flash for the model) that
  runs the exported MLP directly on an MCU, with a built-in self-test mode
  that replays real held-out samples and verifies its predictions match
  the PC/Colab-trained model to float precision.

