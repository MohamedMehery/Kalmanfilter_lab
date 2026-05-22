#include <Arduino.h>

/* ================================================================
 *                    DEBUG & FEATURE TOGGLES
 * ================================================================ */
#define DEBUG_ENABLE        1
#define DEBUG_VERBOSE       1
#define DEBUG_I2C           1

#define FEATURE_LCD         1
#define FEATURE_RTC         0
#define FEATURE_DHT         1
#define FEATURE_KALMAN      1

/* ========== DEBUG MACROS ========== */
#if DEBUG_ENABLE
  #define DEBUG_PRINT(...)      Serial.print(__VA_ARGS__)
  #define DEBUG_PRINTLN(...)    Serial.println(__VA_ARGS__)
  #define DEBUG_PRINTF(...)     do { char _dbg[128]; snprintf(_dbg, sizeof(_dbg), __VA_ARGS__); Serial.print(_dbg); } while(0)
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

#define CHECKPOINT(msg) do { \
    DEBUG_PRINT("[CP] "); \
    DEBUG_PRINT(__LINE__); \
    DEBUG_PRINT(": "); \
    DEBUG_PRINTLN(msg); \
    Serial.flush(); \
    delay(10); \
} while(0)

/* ========== PIN DEFINITIONS ========== */
#define PIN_SDA     PB7
#define PIN_SCL     PB6
#define PIN_DHT     PA0

/* ========== I2C ADDRESSES ========== */
// Standard Wokwi addresses (change if your spec differs)
#define LCD_ADDR    0x27    // Try 0x27 or 0x3F for PCF8574
#define RTC_ADDR    0x68    // Standard DS1307 address

/* ========== I2C TIMING ========== */
// Reduced for faster debugging - increase for spec compliance
#if DEBUG_ENABLE
  #define I2C_DELAY_US  500   // Faster for debug (2kHz)
#else
  #define I2C_DELAY_US  5000  // Spec: 5ms (100Hz per bit)
#endif

/* ========== KALMAN FILTER STRUCTURE ========== */
typedef struct {
    float x;      // State estimate
    float P;      // Estimate uncertainty
    float Q;      // Process noise
    float R;      // Measurement noise
} KalmanFilter_t;

KalmanFilter_t tempFilter;

/* ========== GLOBAL VARIABLES ========== */
float rawTemperature = 25.0f;
float rawHumidity = 50.0f;
float filteredTemperature = 25.0f;

/* ================================================================
 *                    BIT-BANGED I2C IMPLEMENTATION
 * ================================================================ */

void i2c_sda_low(void) {
    pinMode(PIN_SDA, OUTPUT);
    digitalWrite(PIN_SDA, LOW);
}

void i2c_sda_release(void) {
    pinMode(PIN_SDA, INPUT);
}

void i2c_scl_low(void) {
    pinMode(PIN_SCL, OUTPUT);
    digitalWrite(PIN_SCL, LOW);
}

void i2c_scl_release(void) {
    pinMode(PIN_SCL, INPUT);
}

uint8_t i2c_sda_read(void) {
    pinMode(PIN_SDA, INPUT);
    return digitalRead(PIN_SDA);
}

void i2c_delay(void) {
    delayMicroseconds(I2C_DELAY_US);
}

void i2c_init(void) {
    CHECKPOINT("i2c_init: start");
    i2c_sda_release();
    i2c_scl_release();
    i2c_delay();
    CHECKPOINT("i2c_init: done");
}

void i2c_start(void) {
    i2c_sda_release();
    i2c_scl_release();
    i2c_delay();
    i2c_sda_low();
    i2c_delay();
    i2c_scl_low();
    i2c_delay();
}

void i2c_stop(void) {
    i2c_sda_low();
    i2c_scl_release();
    i2c_delay();
    i2c_sda_release();
    i2c_delay();
}

uint8_t i2c_write_byte(uint8_t data) {
    for (int i = 7; i >= 0; i--) {
        if (data & (1 << i)) {
            i2c_sda_release();
        } else {
            i2c_sda_low();
        }
        i2c_delay();
        i2c_scl_release();
        i2c_delay();
        i2c_scl_low();
    }
    
    // Read ACK/NACK
    i2c_sda_release();
    i2c_delay();
    i2c_scl_release();
    i2c_delay();
    uint8_t ack = i2c_sda_read();
    i2c_scl_low();
    i2c_delay();
    
    return ack;  // 0 = ACK, 1 = NACK
}

uint8_t i2c_read_byte(uint8_t send_ack) {
    uint8_t data = 0;
    i2c_sda_release();
    
    for (int i = 7; i >= 0; i--) {
        i2c_scl_release();
        i2c_delay();
        if (i2c_sda_read()) {
            data |= (1 << i);
        }
        i2c_scl_low();
        i2c_delay();
    }
    
    if (send_ack) {
        i2c_sda_release();  // NACK
    } else {
        i2c_sda_low();      // ACK
    }
    i2c_delay();
    i2c_scl_release();
    i2c_delay();
    i2c_scl_low();
    i2c_sda_release();
    i2c_delay();
    
    return data;
}

/* ========== I2C SCAN FUNCTION (DEBUG) ========== */
#if DEBUG_ENABLE
void i2c_scan(void) {
    DEBUG_PRINTLN("I2C Scan starting...");
    int found = 0;
    for (uint8_t addr = 1; addr < 127; addr++) {
        i2c_start();
        uint8_t ack = i2c_write_byte(addr << 1);
        i2c_stop();
        
        if (ack == 0) {
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
 *                    LCD2004 I2C DRIVER
 * ================================================================ */
#if FEATURE_LCD

#define LCD_BACKLIGHT   0x08
#define LCD_ENABLE      0x04
#define LCD_RW          0x02
#define LCD_RS          0x01

bool lcd_write_i2c(uint8_t data) {
    i2c_start();
    uint8_t ack1 = i2c_write_byte(LCD_ADDR << 1);
    uint8_t ack2 = i2c_write_byte(data);
    i2c_stop();
    
    #if DEBUG_I2C
    static uint8_t err_count = 0;
    if (ack1 != 0 || ack2 != 0) {
        if (err_count < 5) {  // Limit error spam
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

void lcd_write_nibble(uint8_t nibble, uint8_t rs) {
    uint8_t data = (nibble & 0xF0) | LCD_BACKLIGHT | (rs ? LCD_RS : 0);
    
    lcd_write_i2c(data | LCD_ENABLE);
    delayMicroseconds(1);
    lcd_write_i2c(data & ~LCD_ENABLE);
    delayMicroseconds(50);
}

void lcd_write_byte(uint8_t data, uint8_t rs) {
    lcd_write_nibble(data & 0xF0, rs);
    lcd_write_nibble((data << 4) & 0xF0, rs);
}

void lcd_command(uint8_t cmd) {
    DEBUG_V_PRINT("  LCD cmd: 0x");
    DEBUG_V_PRINTLN(cmd, HEX);
    lcd_write_byte(cmd, 0);
    if (cmd < 4) delay(2);
}

void lcd_data(uint8_t data) {
    lcd_write_byte(data, 1);
}

bool lcd_init(void) {
    CHECKPOINT("lcd_init: start");
    delay(50);
    
    CHECKPOINT("lcd_init: 4-bit mode sequence");
    lcd_write_nibble(0x30, 0); delay(5);
    CHECKPOINT("lcd_init: nibble 1 done");
    lcd_write_nibble(0x30, 0); delay(1);
    CHECKPOINT("lcd_init: nibble 2 done");
    lcd_write_nibble(0x30, 0); delay(1);
    CHECKPOINT("lcd_init: nibble 3 done");
    lcd_write_nibble(0x20, 0); delay(1);
    CHECKPOINT("lcd_init: 4-bit mode set");
    
    lcd_command(0x28);
    CHECKPOINT("lcd_init: function set done");
    lcd_command(0x0C);
    CHECKPOINT("lcd_init: display on done");
    lcd_command(0x06);
    CHECKPOINT("lcd_init: entry mode done");
    lcd_command(0x01);
    CHECKPOINT("lcd_init: clear done");
    delay(2);
    
    CHECKPOINT("lcd_init: complete!");
    return true;
}

void lcd_set_cursor(uint8_t row, uint8_t col) {
    const uint8_t row_offsets[] = {0x00, 0x40, 0x14, 0x54};
    lcd_command(0x80 | (row_offsets[row] + col));
}

void lcd_print(const char *str) {
    while (*str) {
        lcd_data(*str++);
    }
}

void lcd_clear(void) {
    lcd_command(0x01);
    delay(2);
}

#else
// Stub functions when LCD disabled
bool lcd_init(void) { DEBUG_PRINTLN("[LCD DISABLED]"); return true; }
void lcd_set_cursor(uint8_t row, uint8_t col) { (void)row; (void)col; }
void lcd_print(const char *str) { (void)str; }
void lcd_clear(void) {}
#endif
/* ================================================================
 *                    DHT22 SENSOR DRIVER (FIXED FOR WOKWI)
 * ================================================================ */
#if FEATURE_DHT

bool dht22_read(float *temperature, float *humidity) {
    uint8_t data[5] = {0};
    uint8_t bit_index = 7;
    uint8_t byte_index = 0;
    
    DEBUG_V_PRINTLN("DHT22: Starting read...");
    
    // === IMPROVED START SIGNAL FOR WOKWI ===
    // 1. Pull low for 1-10ms to wake up sensor
    pinMode(PIN_DHT, OUTPUT);
    digitalWrite(PIN_DHT, LOW);
    delay(18);  // Increased to 18ms for better reliability
    
    // 2. Pull high for 20-40us, then release
    digitalWrite(PIN_DHT, HIGH);
    delayMicroseconds(30);  // Brief high pulse
    
    // 3. Switch to input to read sensor response
    pinMode(PIN_DHT, INPUT_PULLUP);  // Use INPUT_PULLUP for better signal
    delayMicroseconds(10);  // Small settling time
    
    // === WAIT FOR DHT22 RESPONSE ===
    // DHT should pull LOW for 80us
    uint32_t timeout = micros() + 100;
    while (digitalRead(PIN_DHT) == HIGH) {
        if (micros() > timeout) {
            DEBUG_V_PRINTLN("DHT22: Timeout waiting for response LOW");
            return false;
        }
    }
    
    // DHT should pull HIGH for 80us
    timeout = micros() + 100;
    while (digitalRead(PIN_DHT) == LOW) {
        if (micros() > timeout) {
            DEBUG_V_PRINTLN("DHT22: Timeout in response LOW period");
            return false;
        }
    }
    
    // Wait for HIGH period to complete
    timeout = micros() + 100;
    while (digitalRead(PIN_DHT) == HIGH) {
        if (micros() > timeout) {
            DEBUG_V_PRINTLN("DHT22: Timeout in response HIGH period");
            return false;
        }
    }
    
    // === READ 40 BITS OF DATA ===
    for (int i = 0; i < 40; i++) {
        // Wait for bit start (LOW period ~50us)
        timeout = micros() + 70;
        while (digitalRead(PIN_DHT) == LOW) {
            if (micros() > timeout) {
                DEBUG_V_PRINT("DHT22: Timeout on bit ");
                DEBUG_V_PRINTLN(i);
                return false;
            }
        }
        
        // Measure HIGH period: 26-28us = 0, 70us = 1
        uint32_t t_start = micros();
        timeout = t_start + 100;
        while (digitalRead(PIN_DHT) == HIGH) {
            if (micros() > timeout) {
                DEBUG_V_PRINT("DHT22: Timeout reading bit ");
                DEBUG_V_PRINTLN(i);
                return false;
            }
        }
        uint32_t t_high = micros() - t_start;
        
        // Threshold: >40us = 1, else = 0
        if (t_high > 40) {
            data[byte_index] |= (1 << bit_index);
        }
        
        if (bit_index == 0) {
            bit_index = 7;
            byte_index++;
        } else {
            bit_index--;
        }
    }
    
    // === CHECKSUM VERIFICATION ===
    uint8_t checksum = data[0] + data[1] + data[2] + data[3];
    if (checksum != data[4]) {
        DEBUG_V_PRINT("DHT22: Checksum mismatch - calc=");
        DEBUG_V_PRINT(checksum);
        DEBUG_V_PRINT(" recv=");
        DEBUG_V_PRINT(data[4]);
        DEBUG_V_PRINT(" [");
        for(int i=0; i<5; i++) {
            DEBUG_V_PRINT(data[i], HEX);
            DEBUG_V_PRINT(" ");
        }
        DEBUG_V_PRINTLN("]");
        return false;
    }
    
    // === PARSE DATA (DHT22 FORMAT) ===
    *humidity = ((data[0] << 8) | data[1]) / 10.0f;
    int16_t temp_raw = ((data[2] & 0x7F) << 8) | data[3];
    if (data[2] & 0x80) temp_raw = -temp_raw;
    *temperature = temp_raw / 10.0f;
    
    DEBUG_V_PRINT("DHT22: SUCCESS! T=");
    DEBUG_V_PRINT(*temperature);
    DEBUG_V_PRINT("C H=");
    DEBUG_V_PRINT(*humidity);
    DEBUG_V_PRINTLN("%");
    
    return true;
}

#else
// Stub - return simulated values
bool dht22_read(float *temperature, float *humidity) {
    static float sim_temp = 25.0f;
    DEBUG_PRINTLN("[DHT DISABLED - using simulated values]");
    
    // Add some realistic variation
    sim_temp += (random(-10, 10) / 100.0f);  // ±0.1°C per read
    if (sim_temp < 20.0f) sim_temp = 20.0f;
    if (sim_temp > 30.0f) sim_temp = 30.0f;
    
    *temperature = sim_temp;
    *humidity = 50.0f + (random(-50, 50) / 10.0f);  // 45-55%
    return true;
}
#endif

/* ================================================================
 *                    KALMAN FILTER
 * ================================================================ */
#if FEATURE_KALMAN

void Kalman_Init(KalmanFilter_t *kf, float initial_estimate, float Q, float R) {
    kf->x = initial_estimate;
    kf->P = 1.0f;
    kf->Q = Q;
    kf->R = R;
    
    DEBUG_PRINTLN("Kalman Filter Initialized:");
    DEBUG_PRINT("  Initial: "); DEBUG_PRINTLN(kf->x);
    DEBUG_PRINT("  Q: "); DEBUG_PRINTLN(kf->Q, 4);
    DEBUG_PRINT("  R: "); DEBUG_PRINTLN(kf->R, 4);
}

float Kalman_Update(KalmanFilter_t *kf, float measurement) {
    float x_pred = kf->x;
    float P_pred = kf->P + kf->Q;
    
    float K = P_pred / (P_pred + kf->R);
    float innovation = measurement - x_pred;
    
    kf->x = x_pred + K * innovation;
    kf->P = (1.0f - K) * P_pred;
    
    DEBUG_V_PRINT("  KF: K=");
    DEBUG_V_PRINT(K, 3);
    DEBUG_V_PRINT(" P=");
    DEBUG_V_PRINTLN(kf->P, 4);
    
    return kf->x;
}

#else
void Kalman_Init(KalmanFilter_t *kf, float initial_estimate, float Q, float R) {
    DEBUG_PRINTLN("[KALMAN DISABLED]");
    kf->x = initial_estimate;
}
float Kalman_Update(KalmanFilter_t *kf, float measurement) {
    return measurement;  // Pass-through
}
#endif

/* ================================================================
 *                    DISPLAY UPDATE
 * ================================================================ */

void update_display(float raw_temp, float filtered_temp, float humidity) {
    #if FEATURE_LCD
    char buf[21];
    
    lcd_set_cursor(0, 0);
    lcd_print("=Kalman Filter Demo=");
    
    lcd_set_cursor(1, 0);
    snprintf(buf, sizeof(buf), "Raw:  %d.%d C      ", 
             (int)raw_temp, abs((int)(raw_temp*10)%10));
    lcd_print(buf);
    
    lcd_set_cursor(2, 0);
    snprintf(buf, sizeof(buf), "Filt: %d.%02d C     ", 
             (int)filtered_temp, abs((int)(filtered_temp*100)%100));
    lcd_print(buf);
    
    lcd_set_cursor(3, 0);
    snprintf(buf, sizeof(buf), "Humidity: %d.%d %%  ", 
             (int)humidity, abs((int)(humidity*10)%10));
    lcd_print(buf);
    #endif
}

/* ================================================================
 *                    SETUP
 * ================================================================ */

void setup() {
    Serial.begin(115200);
    delay(500);
    
    DEBUG_PRINTLN();
    DEBUG_PRINTLN("========================================");
    DEBUG_PRINTLN("  Kalman Filter Tutorial - DEBUG MODE");
    DEBUG_PRINTLN("========================================");
    DEBUG_PRINTLN();
    
    // Print configuration
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
    
    // Initialize I2C
    CHECKPOINT("Initializing I2C...");
    i2c_init();
    delay(100);
    
    // Scan I2C bus (helpful for finding correct addresses!)
    #if DEBUG_ENABLE
    CHECKPOINT("Starting I2C scan...");
    i2c_scan();
    DEBUG_PRINTLN();
    #endif
    
    // Initialize LCD
    #if FEATURE_LCD
    CHECKPOINT("Initializing LCD...");
    lcd_init();
    CHECKPOINT("LCD init returned");
    
    lcd_clear();
    lcd_set_cursor(0, 0);
    lcd_print("Kalman Tutorial");
    lcd_set_cursor(1, 0);
    lcd_print("DEBUG MODE");
    CHECKPOINT("LCD startup text written");
    delay(1000);
    #endif
    
    // Read initial temperature
    CHECKPOINT("Reading initial temperature...");
    float init_temp = 25.0f;
    float init_hum = 50.0f;
    
    for (int attempt = 0; attempt < 3; attempt++) {
        DEBUG_PRINT("DHT read attempt ");
        DEBUG_PRINTLN(attempt + 1);
        if (dht22_read(&init_temp, &init_hum)) {
            DEBUG_PRINT("Success! T=");
            DEBUG_PRINT(init_temp);
            DEBUG_PRINT(" H=");
            DEBUG_PRINTLN(init_hum);
            break;
        }
        delay(500);
    }
    
    // Initialize Kalman filter
    CHECKPOINT("Initializing Kalman filter...");
    Kalman_Init(&tempFilter, init_temp, 0.01f, 0.5f);
    
    rawTemperature = init_temp;
    rawHumidity = init_hum;
    filteredTemperature = init_temp;
    
    #if FEATURE_LCD
    lcd_clear();
    #endif
    
    CHECKPOINT("=== SETUP COMPLETE ===");
    DEBUG_PRINTLN();
    DEBUG_PRINTLN("Time(s)  Raw(C)   Filtered(C)");
    DEBUG_PRINTLN("-------  ------   -----------");
}

/* ================================================================
 *                    MAIN LOOP
 * ================================================================ */

void loop() {
    static unsigned long last_read = 0;
    static unsigned long last_display = 0;
    static unsigned long start_time = millis();
    
    unsigned long now = millis();
    
    // Read DHT22 every 2 seconds
    if (now - last_read >= 2000) {
        last_read = now;
        
        float new_temp, new_hum;
        if (dht22_read(&new_temp, &new_hum)) {
            rawTemperature = new_temp;
            rawHumidity = new_hum;
            
            filteredTemperature = Kalman_Update(&tempFilter, rawTemperature);
            
            // Print to serial
            float elapsed = (now - start_time) / 1000.0f;
            DEBUG_PRINT(elapsed, 1);
            DEBUG_PRINT("     ");
            DEBUG_PRINT(rawTemperature, 2);
            DEBUG_PRINT("     ");
            DEBUG_PRINTLN(filteredTemperature, 2);
        } else {
            DEBUG_PRINTLN("DHT read failed");
        }
    }
    
    // Update display every 500ms
    if (now - last_display >= 500) {
        last_display = now;
        update_display(rawTemperature, filteredTemperature, rawHumidity);
    }
}