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

