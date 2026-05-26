#include <Arduino.h>

/* ================================================================
 *  FILE: main.ino
 *  DESCRIPTION:
 *    Kalman Filter demo on an STM32-class board (Wokwi-compatible).
 *    Reads temperature/humidity from a DHT22 sensor, filters the
 *    temperature reading through an adaptive 1-D Kalman filter, and
 *    displays raw vs. filtered values on a 20×4 I²C LCD.
 *
 *  KALMAN FILTER ENHANCEMENTS IN THIS VERSION:
 *    • Adaptive Q  – process noise grows when the temperature is
 *                    changing quickly (the model becomes less certain
 *                    about its own prediction, so it follows the
 *                    sensor more readily).
 *    • Spike rejection via R inflate – when the observed rate of
 *                    change is physically impossible, the measurement
 *                    noise R is raised dramatically so the filter
 *                    almost ignores the outlier sample.
 *    • dt-aware prediction – the time elapsed between samples is
 *                    measured with millis() inside the filter, so
 *                    the noise model remains consistent even if the
 *                    sampling interval varies.
 *    • Bootstrap initialisation – the very first sample seeds the
 *                    state directly, avoiding a cold-start transient
 *                    that would otherwise take many samples to decay.
 * ================================================================ */

/* ================================================================
 *                    DEBUG & FEATURE TOGGLES
 * ================================================================ */
#define DEBUG_ENABLE        1   // Master switch: 1 = serial debug on
#define DEBUG_VERBOSE       1   // Extra step-by-step Kalman printout
#define DEBUG_I2C           1   // Log I²C ACK/NACK errors

#define FEATURE_LCD         1   // Compile LCD driver
#define FEATURE_RTC         0   // RTC not used in this demo
#define FEATURE_DHT         1   // Compile DHT22 driver
#define FEATURE_KALMAN      1   // Compile Kalman filter

/* ================================================================
 *                    DEBUG MACROS
 *  Using macros (rather than functions) means zero code is emitted
 *  when DEBUG_ENABLE == 0 – the preprocessor strips every call.
 * ================================================================ */
#if DEBUG_ENABLE
  #define DEBUG_PRINT(...)      Serial.print(__VA_ARGS__)
  #define DEBUG_PRINTLN(...)    Serial.println(__VA_ARGS__)
  /* snprintf to a local buffer lets us use printf-style formatting
   * without pulling in the heavy <stdio.h> FILE machinery.         */
  #define DEBUG_PRINTF(...)     do { \
      char _dbg[128]; \
      snprintf(_dbg, sizeof(_dbg), __VA_ARGS__); \
      Serial.print(_dbg); \
  } while(0)
#else
  #define DEBUG_PRINT(...)
  #define DEBUG_PRINTLN(...)
  #define DEBUG_PRINTF(...)
#endif

#if DEBUG_VERBOSE
  #define DEBUG_V_PRINT(...)    Serial.print(__VA_ARGS__)
  #define DEBUG_V_PRINTLN(...)  Serial.println(__VA_ARGS__)
#else
  #define DEBUG_V_PRINT(...)
  #define DEBUG_V_PRINTLN(...)
#endif

#if DEBUG_I2C
  #define DEBUG_I2C_PRINT(...)    Serial.print(__VA_ARGS__)
  #define DEBUG_I2C_PRINTLN(...)  Serial.println(__VA_ARGS__)
#else
  #define DEBUG_I2C_PRINT(...)
  #define DEBUG_I2C_PRINTLN(...)
#endif

/* CHECKPOINT – prints file line number + message, then flushes the
 * TX buffer.  Useful for pinpointing exactly where a hang occurs
 * during hardware bring-up.                                         */
#define CHECKPOINT(msg) do { \
    DEBUG_PRINT("[CP] "); \
    DEBUG_PRINT(__LINE__); \
    DEBUG_PRINT(": "); \
    DEBUG_PRINTLN(msg); \
    Serial.flush(); \
    delay(10); \
} while(0)

/* ================================================================
 *                    PIN DEFINITIONS
 * ================================================================ */
#define PIN_SDA     PB7   // Software I²C data line
#define PIN_SCL     PB6   // Software I²C clock line
#define PIN_DHT     PA0   // DHT22 single-wire data

/* ================================================================
 *                    I²C ADDRESSES
 * ================================================================ */
#define LCD_ADDR    0x27  // PCF8574 I/O expander backing the LCD
#define RTC_ADDR    0x68  // DS1307 (unused here, kept for reference)

/* ================================================================
 *                    I²C BIT-BANG TIMING
 *  The I²C spec requires ≥ 4 µs half-period at 100 kHz.
 *  We use a faster period during debug to reduce serial wait time.
 * ================================================================ */
#if DEBUG_ENABLE
  #define I2C_DELAY_US  500   // ~1 kHz – slow but very tolerant
#else
  #define I2C_DELAY_US  5000  // 100 Hz per bit – spec-compliant
#endif

/* ================================================================
 *              ENHANCED KALMAN FILTER – STRUCT DEFINITION
 *
 *  A 1-D (scalar) discrete Kalman filter tracks a single quantity:
 *  temperature in °C.
 *
 *  Core Kalman variables
 *  ─────────────────────
 *  x   – The current best estimate of the true temperature.
 *         After every update step this is what you use.
 *
 *  P   – The filter's own uncertainty about x, expressed as a
 *         variance (°C²).  A large P means "I am not confident";
 *         a small P means "I am confident".  The filter updates P
 *         automatically every cycle.
 *
 *  Q   – Process noise variance.  Represents how much the true
 *         temperature can change between two consecutive samples
 *         due to real-world physics.  A larger Q makes the filter
 *         more responsive; a smaller Q makes it smoother.
 *         In this enhanced version Q is computed dynamically.
 *
 *  R   – Measurement noise variance (= σ_sensor²).  How noisy the
 *         sensor reading is.  A larger R means "trust the sensor
 *         less"; a smaller R means "trust the sensor more".
 *         In this enhanced version R is inflated for outliers.
 *
 *  Adaptive / configuration variables
 *  ───────────────────────────────────
 *  Q_base        – Minimum Q, representing slow steady-state drift
 *                  (thermal creep when nothing is changing).
 *
 *  Q_max         – Hard ceiling on Q.  Prevents the filter from
 *                  becoming completely unanchored on very fast
 *                  changes.
 *
 *  R_trust       – R used during normal operation (= sensor_σ²).
 *                  Computed once from the sensor's datasheet σ.
 *
 *  R_reject      – Inflated R applied when a spike is detected.
 *                  The Kalman gain K = P/(P+R) collapses toward 0
 *                  when R is huge, so the spike barely moves x.
 *
 *  max_phys_rate – Maximum physically plausible temperature rate
 *                  of change in °C/s.  If the apparent rate from
 *                  the sensor exceeds this, we classify it as a
 *                  spike and apply R_reject.
 *
 *  Bookkeeping
 *  ───────────
 *  last_time     – millis() timestamp of the previous update.
 *                  Enables the filter to compute dt internally
 *                  without burdening the caller.
 *
 *  initialized   – Flag that is 0 before the first measurement.
 *                  On the first call Kalman_Update simply seeds x
 *                  with the measurement value so there is no
 *                  cold-start transient.
 * ================================================================ */
typedef struct {
    float    x;               // Current state estimate (°C)
    float    P;               // State uncertainty / error covariance (°C²)
    float    Q;               // Active process noise (updated each cycle)
    float    R;               // Active measurement noise (updated each cycle)
    float    Q_base;          // Minimum / steady-state process noise
    float    Q_max;           // Upper clamp on Q
    float    R_trust;         // R when sensor reading is trusted
    float    R_reject;        // R when spike is detected (large → ignore)
    float    max_phys_rate;   // Spike threshold in °C/s
    uint32_t last_time;       // millis() at previous Kalman_Update call
    uint8_t  initialized;     // 0 until first sample received
} KalmanFilter_t;

/* One instance for temperature */
KalmanFilter_t tempFilter;

/* ================================================================
 *                    GLOBAL SENSOR STATE
 * ================================================================ */
float rawTemperature    = 25.0f;  // Latest DHT22 temperature (°C)
float rawHumidity       = 50.0f;  // Latest DHT22 humidity    (%)
float filteredTemperature = 25.0f;// Kalman-filtered temperature

/* ================================================================
 *                    BIT-BANGED I²C IMPLEMENTATION
 *
 *  Why bit-bang instead of hardware I²C?
 *  → Gives full control over timing, easier to debug in simulation,
 *    and avoids HAL quirks on different STM32 variants.
 *
 *  I²C protocol recap:
 *    START  – SDA falls while SCL is HIGH  (unique condition)
 *    STOP   – SDA rises while SCL is HIGH  (unique condition)
 *    Data bit – sampled on the rising edge of SCL
 *    ACK    – receiver pulls SDA LOW for the 9th clock pulse
 *    NACK   – SDA remains HIGH for the 9th pulse
 * ================================================================ */

/* Drive SDA LOW by making it an output and writing 0 */
void i2c_sda_low(void)    { pinMode(PIN_SDA, OUTPUT); digitalWrite(PIN_SDA, LOW); }
/* Release SDA – tri-state (input); external pull-up takes it HIGH */
void i2c_sda_release(void){ pinMode(PIN_SDA, INPUT); }
/* Drive SCL LOW */
void i2c_scl_low(void)    { pinMode(PIN_SCL, OUTPUT); digitalWrite(PIN_SCL, LOW); }
/* Release SCL – let pull-up take it HIGH (clock stretching friendly) */
void i2c_scl_release(void){ pinMode(PIN_SCL, INPUT); }
/* Sample SDA */
uint8_t i2c_sda_read(void){ pinMode(PIN_SDA, INPUT); return digitalRead(PIN_SDA); }
/* Half-period delay – determines bus speed */
void i2c_delay(void)      { delayMicroseconds(I2C_DELAY_US); }

void i2c_init(void) {
    CHECKPOINT("i2c_init: start");
    i2c_sda_release();   // Both lines idle HIGH
    i2c_scl_release();
    i2c_delay();
    CHECKPOINT("i2c_init: done");
}

/* START condition: SDA goes LOW while SCL is still HIGH */
void i2c_start(void) {
    i2c_sda_release();
    i2c_scl_release();
    i2c_delay();
    i2c_sda_low();   // ← the START event
    i2c_delay();
    i2c_scl_low();   // SCL low: prepare for first data bit
    i2c_delay();
}

/* STOP condition: SDA goes HIGH while SCL is HIGH */
void i2c_stop(void) {
    i2c_sda_low();
    i2c_scl_release();
    i2c_delay();
    i2c_sda_release(); // ← the STOP event
    i2c_delay();
}

/* Write one byte MSB-first.  Returns 0 if slave ACKed, 1 if NACK. */
uint8_t i2c_write_byte(uint8_t data) {
    for (int i = 7; i >= 0; i--) {
        /* Present each bit on SDA before rising clock edge */
        if (data & (1 << i)) i2c_sda_release();
        else                  i2c_sda_low();
        i2c_delay();
        i2c_scl_release();  // Rising edge – slave latches the bit
        i2c_delay();
        i2c_scl_low();      // Falling edge – master may change SDA
    }
    /* 9th clock: master releases SDA, slave drives ACK (LOW) */
    i2c_sda_release();
    i2c_delay();
    i2c_scl_release();
    i2c_delay();
    uint8_t ack = i2c_sda_read();  // 0 = ACK, 1 = NACK
    i2c_scl_low();
    i2c_delay();
    return ack;
}

/* Read one byte MSB-first.
 * send_ack = 0 → send ACK  (more bytes will follow)
 * send_ack = 1 → send NACK (this is the last byte)           */
uint8_t i2c_read_byte(uint8_t send_ack) {
    uint8_t data = 0;
    i2c_sda_release();  // Release SDA so slave can drive it
    for (int i = 7; i >= 0; i--) {
        i2c_scl_release();
        i2c_delay();
        if (i2c_sda_read()) data |= (1 << i);  // Sample on rising edge
        i2c_scl_low();
        i2c_delay();
    }
    /* Send ACK or NACK on the 9th clock */
    if (send_ack) i2c_sda_release();  // NACK – we are done reading
    else          i2c_sda_low();      // ACK  – slave should send more
    i2c_delay();
    i2c_scl_release();
    i2c_delay();
    i2c_scl_low();
    i2c_sda_release();
    i2c_delay();
    return data;
}

/* ========== I²C BUS SCANNER ========== */
#if DEBUG_ENABLE
void i2c_scan(void) {
    DEBUG_PRINTLN("I2C Scan starting...");
    int found = 0;
    /* The 7-bit address space is 1–126.  Address 0 is the general
     * call address and 127 is reserved.                            */
    for (uint8_t addr = 1; addr < 127; addr++) {
        i2c_start();
        uint8_t ack = i2c_write_byte(addr << 1); // Write mode (R/W=0)
        i2c_stop();
        if (ack == 0) {   // Slave pulled SDA low → device present
            DEBUG_PRINT("  Found device at 0x");
            DEBUG_PRINTLN(addr, HEX);
            found++;
        }
        delay(5);
    }
    DEBUG_PRINT("Scan complete. Devices found: ");
    DEBUG_PRINTLN(found);
}
#endif

/* ================================================================
 *                    LCD2004 I²C DRIVER
 *
 *  The LCD is wired through a PCF8574 GPIO expander.
 *  The expander's output byte maps to LCD control pins as follows:
 *    Bit 7-4 : D7-D4  (4-bit data nibble)
 *    Bit 3   : BL     (backlight transistor)
 *    Bit 2   : EN     (enable – latches data on falling edge)
 *    Bit 1   : RW     (0 = write, 1 = read; keep 0 for write-only)
 *    Bit 0   : RS     (0 = command register, 1 = data register)
 *
 *  The HD44780 uses a two-nibble (4-bit) serial protocol:
 *    1. Send upper nibble with EN pulse
 *    2. Send lower nibble with EN pulse
 * ================================================================ */
#if FEATURE_LCD

#define LCD_BACKLIGHT   0x08  // Bit 3 of the expander byte
#define LCD_ENABLE      0x04  // Bit 2
#define LCD_RW          0x02  // Bit 1 (always 0 for write)
#define LCD_RS          0x01  // Bit 0

/* Write one raw byte to the PCF8574 over I²C */
bool lcd_write_i2c(uint8_t data) {
    i2c_start();
    uint8_t ack1 = i2c_write_byte(LCD_ADDR << 1);  // Address + write
    uint8_t ack2 = i2c_write_byte(data);             // Payload
    i2c_stop();
#if DEBUG_I2C
    static uint8_t err_count = 0;
    if (ack1 != 0 || ack2 != 0) {
        if (err_count < 5) {
            DEBUG_I2C_PRINT("LCD I2C error: addr_ack=");
            DEBUG_I2C_PRINT(ack1);
            DEBUG_I2C_PRINT(" data_ack=");
            DEBUG_I2C_PRINTLN(ack2);
            err_count++;
        }
        return false;
    }
#endif
    return true;
}

/* Send a 4-bit nibble to the LCD.
 * rs = 0 → command register   rs = 1 → data (character) register  */
void lcd_write_nibble(uint8_t nibble, uint8_t rs) {
    /* Combine nibble bits with the permanent control lines */
    uint8_t data = (nibble & 0xF0)
                 | LCD_BACKLIGHT
                 | (rs ? LCD_RS : 0);

    lcd_write_i2c(data | LCD_ENABLE);   // EN high – setup
    delayMicroseconds(1);
    lcd_write_i2c(data & ~LCD_ENABLE);  // EN low  – latch
    delayMicroseconds(50);              // HD44780 needs ≥ 37 µs
}

/* Send a full byte as two nibbles (high nibble first) */
void lcd_write_byte(uint8_t data, uint8_t rs) {
    lcd_write_nibble( data        & 0xF0, rs);  // Upper nibble
    lcd_write_nibble((data << 4)  & 0xF0, rs);  // Lower nibble
}

void lcd_command(uint8_t cmd) {
    DEBUG_V_PRINT("  LCD cmd: 0x");
    DEBUG_V_PRINTLN(cmd, HEX);
    lcd_write_byte(cmd, 0);         // RS = 0 → instruction register
    if (cmd < 4) delay(2);          // Clear / home need ≥ 1.52 ms
}

void lcd_data(uint8_t data) {
    lcd_write_byte(data, 1);        // RS = 1 → DDRAM / character
}

bool lcd_init(void) {
    CHECKPOINT("lcd_init: start");
    delay(50);  // VCC rise time – HD44780 needs > 40 ms after power-on

    /* Software reset sequence (4-bit mode initialisation)
     * Three 0x30 nibbles put the controller into a known state
     * regardless of whether it was previously in 4-bit or 8-bit mode */
    CHECKPOINT("lcd_init: 4-bit mode sequence");
    lcd_write_nibble(0x30, 0); delay(5);
    lcd_write_nibble(0x30, 0); delay(1);
    lcd_write_nibble(0x30, 0); delay(1);
    lcd_write_nibble(0x20, 0); delay(1);  // Switch to 4-bit mode
    CHECKPOINT("lcd_init: 4-bit mode set");

    lcd_command(0x28);  // Function set: 4-bit, 2 lines, 5×8 font
    CHECKPOINT("lcd_init: function set done");
    lcd_command(0x0C);  // Display ON, cursor OFF, blink OFF
    CHECKPOINT("lcd_init: display on done");
    lcd_command(0x06);  // Entry mode: increment cursor, no display shift
    CHECKPOINT("lcd_init: entry mode done");
    lcd_command(0x01);  // Clear display (also resets cursor to home)
    CHECKPOINT("lcd_init: clear done");
    delay(2);

    CHECKPOINT("lcd_init: complete!");
    return true;
}

/* Move cursor to (row, col) – rows 0-3, columns 0-19 for a 20×4 LCD */
void lcd_set_cursor(uint8_t row, uint8_t col) {
    /* DDRAM addresses for a 20×4 display:
     *   Row 0: 0x00-0x13   Row 1: 0x40-0x53
     *   Row 2: 0x14-0x27   Row 3: 0x54-0x67   */
    const uint8_t row_offsets[] = {0x00, 0x40, 0x14, 0x54};
    lcd_command(0x80 | (row_offsets[row] + col));
}

void lcd_print(const char *str) {
    while (*str) lcd_data(*str++);
}

void lcd_clear(void) {
    lcd_command(0x01);
    delay(2);
}

#else
/* ---- Stub functions when LCD feature is disabled ---- */
bool lcd_init(void)                       { DEBUG_PRINTLN("[LCD DISABLED]"); return true; }
void lcd_set_cursor(uint8_t r, uint8_t c) { (void)r; (void)c; }
void lcd_print(const char *s)             { (void)s; }
void lcd_clear(void)                      {}
#endif  /* FEATURE_LCD */

/* ================================================================
 *                    DHT22 SENSOR DRIVER
 *
 *  The DHT22 uses a proprietary single-wire protocol:
 *   1. MCU sends a start pulse (LOW ≥ 1 ms, then release)
 *   2. Sensor responds: 80 µs LOW, 80 µs HIGH
 *   3. 40 data bits follow, each encoded as:
 *        50 µs LOW preamble  +  26-28 µs HIGH → bit '0'
 *        50 µs LOW preamble  +  70 µs    HIGH → bit '1'
 *   4. Bits are arranged as: RH high, RH low, T high, T low, CRC
 * ================================================================ */
#if FEATURE_DHT

bool dht22_read(float *temperature, float *humidity) {
    uint8_t data[5]  = {0};
    uint8_t bit_idx  = 7;
    uint8_t byte_idx = 0;

    DEBUG_V_PRINTLN("DHT22: Starting read...");

    /* --- Step 1: Send start pulse ---
     * Pull the bus LOW for ≥ 1 ms to wake the sensor (18 ms is safe). */
    pinMode(PIN_DHT, OUTPUT);
    digitalWrite(PIN_DHT, LOW);
    delay(18);

    /* Brief HIGH then release: sensor takes over */
    digitalWrite(PIN_DHT, HIGH);
    delayMicroseconds(30);
    pinMode(PIN_DHT, INPUT_PULLUP);
    delayMicroseconds(10);

    /* --- Step 2: Wait for sensor response ---
     * Expect: ~80 µs LOW, ~80 µs HIGH from the DHT22.              */
    uint32_t timeout;

    timeout = micros() + 100;
    while (digitalRead(PIN_DHT) == HIGH) {
        if (micros() > timeout) { DEBUG_V_PRINTLN("DHT22: No response LOW"); return false; }
    }
    timeout = micros() + 100;
    while (digitalRead(PIN_DHT) == LOW) {
        if (micros() > timeout) { DEBUG_V_PRINTLN("DHT22: Stuck in response LOW"); return false; }
    }
    timeout = micros() + 100;
    while (digitalRead(PIN_DHT) == HIGH) {
        if (micros() > timeout) { DEBUG_V_PRINTLN("DHT22: Stuck in response HIGH"); return false; }
    }

    /* --- Step 3: Read 40 data bits --- */
    for (int i = 0; i < 40; i++) {
        /* Each bit starts with a 50 µs LOW preamble */
        timeout = micros() + 70;
        while (digitalRead(PIN_DHT) == LOW) {
            if (micros() > timeout) {
                DEBUG_V_PRINT("DHT22: LOW timeout bit "); DEBUG_V_PRINTLN(i);
                return false;
            }
        }

        /* Measure how long the HIGH phase lasts:
         *   < 40 µs → '0'    ≥ 40 µs → '1'                        */
        uint32_t t_start = micros();
        timeout = t_start + 100;
        while (digitalRead(PIN_DHT) == HIGH) {
            if (micros() > timeout) {
                DEBUG_V_PRINT("DHT22: HIGH timeout bit "); DEBUG_V_PRINTLN(i);
                return false;
            }
        }
        if ((micros() - t_start) > 40) {
            data[byte_idx] |= (1 << bit_idx);
        }

        /* Advance bit pointer, wrapping MSB→LSB across bytes */
        if (bit_idx == 0) { bit_idx = 7; byte_idx++; }
        else              { bit_idx--; }
    }

    /* --- Step 4: Verify checksum ---
     * The 5th byte is the 8-bit sum of the first four.             */
    uint8_t checksum = data[0] + data[1] + data[2] + data[3];
    if (checksum != data[4]) {
        DEBUG_V_PRINT("DHT22: Checksum fail – calc=0x"); DEBUG_V_PRINT(checksum, HEX);
        DEBUG_V_PRINT(" recv=0x"); DEBUG_V_PRINTLN(data[4], HEX);
        return false;
    }

    /* --- Step 5: Decode ---
     * Humidity    = (byte0 << 8 | byte1) / 10   (unsigned)
     * Temperature = (byte2[6:0] << 8 | byte3) / 10
     *               Sign bit is byte2 bit 7 (1 = negative)         */
    *humidity    = ((data[0] << 8) | data[1]) / 10.0f;
    int16_t t_raw = ((data[2] & 0x7F) << 8) | data[3];
    if (data[2] & 0x80) t_raw = -t_raw;
    *temperature = t_raw / 10.0f;

    DEBUG_V_PRINT("DHT22: T="); DEBUG_V_PRINT(*temperature);
    DEBUG_V_PRINT("°C  H="); DEBUG_V_PRINT(*humidity); DEBUG_V_PRINTLN("%");
    return true;
}

#else
/* Simulated sensor used when FEATURE_DHT == 0 */
bool dht22_read(float *temperature, float *humidity) {
    static float sim_temp = 25.0f;
    DEBUG_PRINTLN("[DHT DISABLED – simulated values]");
    sim_temp += (random(-10, 10) / 100.0f);
    sim_temp  = constrain(sim_temp, 20.0f, 30.0f);
    *temperature = sim_temp;
    *humidity    = 50.0f + (random(-50, 50) / 10.0f);
    return true;
}
#endif  /* FEATURE_DHT */

/* ================================================================
 *                    ENHANCED KALMAN FILTER IMPLEMENTATION
 * ================================================================ */
#if FEATURE_KALMAN

/* ----------------------------------------------------------------
 *  Kalman_Init
 *  -----------
 *  Parameters
 *    kf            – Pointer to the filter instance to initialise.
 *    sensor_sigma  – Standard deviation of the sensor noise (°C).
 *                    For the DHT22 the datasheet gives ±0.5 °C,
 *                    so pass 0.5f.  We store σ² = variance.
 *    max_phys_rate – The fastest temperature change the process
 *                    can physically produce in °C/s.
 *                    Example: 1.0 for a slow HVAC room,
 *                             3.0 for a kettle or engine bay.
 *                    Any sensor jump exceeding this rate is
 *                    treated as a spike.
 * ---------------------------------------------------------------- */
void Kalman_Init(KalmanFilter_t *kf, float sensor_sigma, float max_phys_rate) {
    kf->x            = 0.0f;   // Will be overwritten by first measurement
    kf->P            = 1.0f;   // Initial uncertainty: ±1 °C² (broad prior)

    /* Q_base – the irreducible background drift between samples.
     * 1e-4 °C²/s is a reasonable guess for slow indoor temperature. */
    kf->Q_base       = 1e-4f;

    /* Q_max – caps Q so the filter never becomes completely noisy */
    kf->Q_max        = 1.0f;

    /* R_trust – normal sensor variance (σ²).  The filter trusts the
     * sensor this much when the measurement is physically plausible. */
    kf->R_trust      = sensor_sigma * sensor_sigma;

    /* R_reject – used for spikes.  100× variance means the Kalman
     * gain K collapses to near zero, so the filter barely moves.   */
    kf->R_reject     = 100.0f * kf->R_trust;

    /* Set working values to their default (normal) settings */
    kf->Q            = kf->Q_base;
    kf->R            = kf->R_trust;

    kf->max_phys_rate = max_phys_rate;
    kf->last_time    = 0;
    kf->initialized  = 0;  // Force bootstrap on first update call

    DEBUG_PRINTLN("Kalman Filter Initialized:");
    DEBUG_PRINT("  sensor_sigma  : "); DEBUG_PRINTLN(sensor_sigma, 4);
    DEBUG_PRINT("  R_trust       : "); DEBUG_PRINTLN(kf->R_trust,  4);
    DEBUG_PRINT("  R_reject      : "); DEBUG_PRINTLN(kf->R_reject, 4);
    DEBUG_PRINT("  Q_base        : "); DEBUG_PRINTLN(kf->Q_base,   6);
    DEBUG_PRINT("  Q_max         : "); DEBUG_PRINTLN(kf->Q_max,    4);
    DEBUG_PRINT("  max_phys_rate : "); DEBUG_PRINT(kf->max_phys_rate, 2);
    DEBUG_PRINTLN(" °C/s");
}

/* ----------------------------------------------------------------
 *  Kalman_Tune  (internal helper – not called by application code)
 *  ------------
 *  Adapts Q and R before each predict/update step.
 *
 *  Logic:
 *    1. Compute the observed rate of change of the measurement
 *       relative to the current estimate: |z - x| / dt.
 *
 *    2. If that rate exceeds max_phys_rate → SPIKE
 *         • Inflate R to R_reject so the update barely affects x.
 *         • Keep Q at Q_base so the model stays confident.
 *
 *    3. Otherwise → VALID CHANGE
 *         • Use R_trust (normal confidence in the sensor).
 *         • Q scales with (apparent_rate²) × dt so that a fast
 *           but real change widens P_pred, allowing the filter to
 *           follow more aggressively.
 *
 *  Why quadratic Q?
 *    Variance ∝ (rate × time)².  If the temperature is changing at
 *    1 °C/s over a 2 s window, the model uncertainty should be
 *    roughly (1×2)² = 4 °C², not just a constant.
 * ---------------------------------------------------------------- */
static void Kalman_Tune(KalmanFilter_t *kf, float measurement, float dt_s) {
    /* Protect against zero or negative dt (clock wrap, first call) */
    if (dt_s < 1e-3f) dt_s = 1e-3f;

    float innovation     = fabsf(measurement - kf->x);   // |z − x̂|
    float apparent_rate  = innovation / dt_s;              // °C/s

    if (apparent_rate > kf->max_phys_rate) {
        /* ---- SPIKE DETECTED ---- */
        kf->R = kf->R_reject;  // Distrust the sensor heavily
        kf->Q = kf->Q_base;    // Model stays steady (nothing real happened)

        DEBUG_V_PRINT("  KF TUNE: SPIKE  rate=");
        DEBUG_V_PRINT(apparent_rate, 2);
        DEBUG_V_PRINT("°C/s  R→"); DEBUG_V_PRINTLN(kf->R, 2);
    } else {
        /* ---- PLAUSIBLE CHANGE ---- */
        kf->R = kf->R_trust;

        /* Compute adaptive Q: base noise + variance due to the
         * observed rate of change (quadratic growth with speed)    */
        float q = kf->Q_base + (apparent_rate * apparent_rate) * dt_s;
        if (q > kf->Q_max) q = kf->Q_max;  // Hard ceiling
        kf->Q = q;

        DEBUG_V_PRINT("  KF TUNE: OK     rate=");
        DEBUG_V_PRINT(apparent_rate, 2);
        DEBUG_V_PRINT("°C/s  Q=");
        DEBUG_V_PRINT(kf->Q, 5);
        DEBUG_V_PRINT("  R="); DEBUG_V_PRINTLN(kf->R, 4);
    }
}

/* ----------------------------------------------------------------
 *  Kalman_Update
 *  -------------
 *  Call once per new sensor reading.  Returns the filtered estimate.
 *
 *  Discrete 1-D Kalman equations used here:
 *
 *    PREDICT step (project state and covariance forward in time)
 *      x_pred = x                (constant-temperature process model)
 *      P_pred = P + Q            (uncertainty grows by process noise)
 *
 *    UPDATE step (fuse prediction with new measurement z)
 *      K   = P_pred / (P_pred + R)     ← Kalman gain (0…1)
 *      x   = x_pred + K × (z − x_pred) ← weighted average
 *      P   = (1 − K) × P_pred           ← reduced uncertainty
 *
 *  Interpretation of K:
 *    K → 0 : R large  → sensor is noisy, ignore measurement → smooth
 *    K → 1 : P large  → model is uncertain, trust sensor → responsive
 * ---------------------------------------------------------------- */
float Kalman_Update(KalmanFilter_t *kf, float measurement) {
    uint32_t now = millis();

    /* --- Bootstrap: seed filter on very first sample ---
     * Without this the initial x = 0, which would produce a huge
     * spurious innovation on the first real measurement, causing the
     * Tune function to wrongly classify it as a spike.              */
    if (!kf->initialized) {
        kf->x           = measurement;
        kf->P           = 1.0f;        // Reset to broad initial uncertainty
        kf->last_time   = now;
        kf->initialized = 1;
        DEBUG_V_PRINT("  KF Bootstrap: x0=");
        DEBUG_V_PRINTLN(kf->x, 2);
        return kf->x;
    }

    /* Compute elapsed time since last update in seconds */
    float dt_s = (float)(now - kf->last_time) / 1000.0f;
    kf->last_time = now;

    /* Adapt Q and R before running the filter equations */
    Kalman_Tune(kf, measurement, dt_s);

    /* ---- PREDICT ---- */
    float P_pred = kf->P + kf->Q;   // Covariance grows (less certain)

    /* ---- UPDATE ---- */
    float K   = P_pred / (P_pred + kf->R);          // Kalman gain
    float innov = measurement - kf->x;               // Innovation (residual)
    kf->x     = kf->x + K * innov;                  // Fused estimate
    kf->P     = (1.0f - K) * P_pred;                // Updated covariance

    DEBUG_V_PRINT("  KF UPDATE: K=");
    DEBUG_V_PRINT(K, 4);
    DEBUG_V_PRINT("  innov=");
    DEBUG_V_PRINT(innov, 3);
    DEBUG_V_PRINT("  x=");
    DEBUG_V_PRINT(kf->x, 3);
    DEBUG_V_PRINT("  P=");
    DEBUG_V_PRINTLN(kf->P, 5);

    return kf->x;
}

#else
/* ---- Stubs when FEATURE_KALMAN == 0 (pass-through) ---- */
void  Kalman_Init  (KalmanFilter_t *kf, float s, float r) {
    DEBUG_PRINTLN("[KALMAN DISABLED]"); kf->x = 0; kf->initialized = 0;
    (void)s; (void)r;
}
float Kalman_Update(KalmanFilter_t *kf, float z)          { return z; (void)kf; }
#endif  /* FEATURE_KALMAN */

/* ================================================================
 *                    DISPLAY UPDATE
 *
 *  Refreshes all four lines of the LCD with the latest readings.
 *  snprintf is used to format floats manually because printf("%f")
 *  is often disabled on AVR/STM32 to save flash.
 * ================================================================ */
void update_display(float raw_temp, float filtered_temp, float humidity) {
#if FEATURE_LCD
    char buf[21];   // 20 chars + null terminator

    lcd_set_cursor(0, 0);
    lcd_print("=Kalman Filter Demo=");

    /* Manual fixed-point formatting: cast integer part, extract one
     * decimal by taking (value*10) % 10 and abs() to handle negatives */
    lcd_set_cursor(1, 0);
    snprintf(buf, sizeof(buf), "Raw:  %d.%d C      ",
             (int)raw_temp,
             abs((int)(raw_temp * 10) % 10));
    lcd_print(buf);

    lcd_set_cursor(2, 0);
    snprintf(buf, sizeof(buf), "Filt: %d.%02d C     ",
             (int)filtered_temp,
             abs((int)(filtered_temp * 100) % 100));
    lcd_print(buf);

    lcd_set_cursor(3, 0);
    snprintf(buf, sizeof(buf), "Humidity: %d.%d %%  ",
             (int)humidity,
             abs((int)(humidity * 10) % 10));
    lcd_print(buf);
#endif
}

/* ================================================================
 *                    SETUP
 * ================================================================ */
void setup() {
    Serial.begin(115200);
    delay(500);  // Let the USB/UART settle

    DEBUG_PRINTLN();
    DEBUG_PRINTLN("========================================");
    DEBUG_PRINTLN("  Kalman Filter Tutorial - DEBUG MODE  ");
    DEBUG_PRINTLN("========================================");
    DEBUG_PRINTLN();

    /* Print compile-time feature configuration */
    DEBUG_PRINTLN("Configuration:");
    DEBUG_PRINT("  DEBUG_ENABLE:   "); DEBUG_PRINTLN(DEBUG_ENABLE);
    DEBUG_PRINT("  DEBUG_VERBOSE:  "); DEBUG_PRINTLN(DEBUG_VERBOSE);
    DEBUG_PRINT("  DEBUG_I2C:      "); DEBUG_PRINTLN(DEBUG_I2C);
    DEBUG_PRINT("  FEATURE_LCD:    "); DEBUG_PRINTLN(FEATURE_LCD);
    DEBUG_PRINT("  FEATURE_RTC:    "); DEBUG_PRINTLN(FEATURE_RTC);
    DEBUG_PRINT("  FEATURE_DHT:    "); DEBUG_PRINTLN(FEATURE_DHT);
    DEBUG_PRINT("  FEATURE_KALMAN: "); DEBUG_PRINTLN(FEATURE_KALMAN);
    DEBUG_PRINT("  I2C_DELAY_US:   "); DEBUG_PRINTLN(I2C_DELAY_US);
    DEBUG_PRINT("  LCD_ADDR:       0x"); DEBUG_PRINTLN(LCD_ADDR, HEX);
    DEBUG_PRINTLN();

    /* ---- I²C ---- */
    CHECKPOINT("Initializing I2C...");
    i2c_init();
    delay(100);

#if DEBUG_ENABLE
    CHECKPOINT("Starting I2C scan...");
    i2c_scan();
    DEBUG_PRINTLN();
#endif

    /* ---- LCD ---- */
#if FEATURE_LCD
    CHECKPOINT("Initializing LCD...");
    lcd_init();
    CHECKPOINT("LCD init returned");
    lcd_clear();
    lcd_set_cursor(0, 0); lcd_print("Kalman Tutorial");
    lcd_set_cursor(1, 0); lcd_print("DEBUG MODE");
    CHECKPOINT("LCD startup text written");
    delay(1000);
#endif

    /* ---- Initial sensor read (used to seed the Kalman filter) ---- */
    CHECKPOINT("Reading initial temperature...");
    float init_temp = 25.0f;  // Safe fallback if DHT read fails
    float init_hum  = 50.0f;

    for (int attempt = 0; attempt < 3; attempt++) {
        DEBUG_PRINT("DHT read attempt "); DEBUG_PRINTLN(attempt + 1);
        if (dht22_read(&init_temp, &init_hum)) {
            DEBUG_PRINT("Success! T="); DEBUG_PRINT(init_temp);
            DEBUG_PRINT(" H="); DEBUG_PRINTLN(init_hum);
            break;
        }
        delay(500);
    }

    /* ---- Kalman filter initialisation ----
     *
     *  sensor_sigma  = 0.5f
     *    The DHT22 datasheet specifies ±0.5 °C accuracy.
     *    We pass σ; the filter squares it internally to get R = 0.25.
     *
     *  max_phys_rate = 1.0f  (°C/s)
     *    For an indoor temperature sensor 1 °C/s is a generous upper
     *    bound.  Any jump larger than this per second is a sensor
     *    glitch and will be rejected via R inflation.
     *
     *  NOTE: The new Kalman_Init no longer takes an initial estimate.
     *  The filter's Kalman_Update bootstraps itself on the very first
     *  measurement it receives, eliminating the cold-start transient.
     */
    CHECKPOINT("Initializing Kalman filter...");
    Kalman_Init(&tempFilter, 0.5f, 1.0f);

    /* Pre-seed global state with the initial sensor reading */
    rawTemperature      = init_temp;
    rawHumidity         = init_hum;
    filteredTemperature = init_temp;

#if FEATURE_LCD
    lcd_clear();
#endif

    CHECKPOINT("=== SETUP COMPLETE ===");
    DEBUG_PRINTLN();
    DEBUG_PRINTLN("Time(s)  Raw(C)   Filtered(C)  K-Gain");
    DEBUG_PRINTLN("-------  ------   -----------  ------");
}

/* ================================================================
 *                    MAIN LOOP
 * ================================================================ */
void loop() {
    static unsigned long last_read    = 0;
    static unsigned long last_display = 0;
    static unsigned long start_time   = millis();

    unsigned long now = millis();

    /* ---- Read sensor every 2 seconds ---- */
    if (now - last_read >= 2000) {
        last_read = now;

        float new_temp, new_hum;
        if (dht22_read(&new_temp, &new_hum)) {
            rawTemperature = new_temp;
            rawHumidity    = new_hum;

            /* Pass raw reading into the Kalman filter.
             * Kalman_Update handles timing internally via millis().
             * The returned value is the filtered (smoothed) estimate. */
            filteredTemperature = Kalman_Update(&tempFilter, rawTemperature);

            /* ---- Serial log (CSV-friendly for plotting) ---- */
            float elapsed = (now - start_time) / 1000.0f;

            /* Compute Kalman gain for display purposes:
             *   K = P_pred / (P_pred + R)  where P_pred ≈ P + Q
             * This is a read-only approximation for logging only;
             * the filter has already updated its internal P.         */
            float P_pred_approx  = tempFilter.P + tempFilter.Q;
            float K_approx       = P_pred_approx / (P_pred_approx + tempFilter.R);

            DEBUG_PRINT(elapsed,            1); DEBUG_PRINT("     ");
            DEBUG_PRINT(rawTemperature,     2); DEBUG_PRINT("     ");
            DEBUG_PRINT(filteredTemperature,2); DEBUG_PRINT("       ");
            DEBUG_PRINTLN(K_approx,         4);

        } else {
            DEBUG_PRINTLN("DHT read failed – skipping Kalman update");
        }
    }

    /* ---- Refresh LCD every 200 ms ---- */
    if (now - last_display >= 200) {
        last_display = now;
        update_display(rawTemperature, filteredTemperature, rawHumidity);
    }
}