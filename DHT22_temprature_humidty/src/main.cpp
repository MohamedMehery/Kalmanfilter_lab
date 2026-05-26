#include <Arduino.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ================================================================
 *  FILE: main.ino
 *  DESCRIPTION:
 *    Scientific testbench for comparing a FIXED-parameter Kalman
 *    filter against an ADAPTIVE Kalman filter for temperature
 *    estimation.
 *
 *    Two modes of operation are selectable at compile time:
 *      FEATURE_TESTBENCH = 1   → Scripted ground-truth + synthetic
 *                                noise + spike injection.  Both
 *                                filters consume the same input.
 *                                CSV is streamed over the serial
 *                                port and a RMSE table is printed
 *                                when the run completes.
 *      FEATURE_TESTBENCH = 0   → Original live demo using DHT22.
 *
 *    The testbench is intentionally deterministic (fixed PRNG seed)
 *    so results are reproducible across runs and can be cited in a
 *    scientific report.
 * ================================================================ */

/* ================================================================
 *                    DEBUG & FEATURE TOGGLES
 * ================================================================ */
#define DEBUG_ENABLE        1
#define DEBUG_VERBOSE       0   // Off in testbench – CSV must stay clean
#define DEBUG_I2C           0

#define FEATURE_LCD         1
#define FEATURE_RTC         0
#define FEATURE_DHT         1
#define FEATURE_KALMAN      1
#define FEATURE_TESTBENCH   0   // <<< MAIN SWITCH (1 = science mode)

/* ================================================================
 *                    TESTBENCH CONFIGURATION
 * ================================================================ */
#define TEST_SAMPLE_MS      250      // 4 Hz sampling → 1680 samples
#define TEST_DURATION_S     420.0f   // total experiment length
#define TEST_RANDOM_SEED    12345UL  // fixed PRNG seed (reproducible)
#define TEST_NOISE_SIGMA    0.5f     // σ of additive Gaussian noise (°C)

#define TEST_SPIKE_TIME_S   260.0f   // when to inject the outlier
#define TEST_SPIKE_VALUE    55.0f    // forced measurement value

/* Fixed-Kalman tuning – a reasonable compromise that a non-adaptive
 * implementation might use.  These values are intentionally NOT tuned
 * per-scenario; that is the whole point of the comparison.            */
#define FIXED_KF_Q          0.01f
#define FIXED_KF_R          0.25f    // = sensor_sigma² = 0.5² = 0.25

/* Adaptive-Kalman tuning */
#define ADAPT_KF_SIGMA      0.5f
#define ADAPT_KF_MAXRATE    1.0f     // °C/s plausibility limit

/* ================================================================
 *                    DEBUG MACROS
 * ================================================================ */
#if DEBUG_ENABLE
  #define DEBUG_PRINT(...)      Serial.print(__VA_ARGS__)
  #define DEBUG_PRINTLN(...)    Serial.println(__VA_ARGS__)
#else
  #define DEBUG_PRINT(...)
  #define DEBUG_PRINTLN(...)
#endif

#if DEBUG_VERBOSE
  #define DEBUG_V_PRINT(...)    Serial.print(__VA_ARGS__)
  #define DEBUG_V_PRINTLN(...)  Serial.println(__VA_ARGS__)
#else
  #define DEBUG_V_PRINT(...)
  #define DEBUG_V_PRINTLN(...)
#endif

#define CHECKPOINT(msg) do { \
    DEBUG_PRINT("# [CP] "); DEBUG_PRINT(__LINE__); \
    DEBUG_PRINT(": ");      DEBUG_PRINTLN(msg); \
    Serial.flush(); delay(10); \
} while(0)

/* ================================================================
 *                    PIN / I²C DEFINITIONS
 * ================================================================ */
#define PIN_SDA     PB7
#define PIN_SCL     PB6
#define PIN_DHT     PA0
#define LCD_ADDR    0x27
#define I2C_DELAY_US 500

/* ================================================================
 *              ADAPTIVE KALMAN FILTER STRUCT
 *  (Extended with instrumentation fields for the testbench.)
 * ================================================================ */
typedef struct {
    float    x;
    float    P;
    float    Q;
    float    R;
    float    Q_base;
    float    Q_max;
    float    R_trust;
    float    R_reject;
    float    max_phys_rate;
    uint32_t last_time;
    uint8_t  initialized;

    /* ---- Instrumentation (telemetry for CSV log / analysis) ---- */
    float    last_K;         // Kalman gain from most recent update
    float    last_rate;      // observed |Δz|/Δt that drove tuning
    float    last_innov;     // innovation (z - x_pred)
    uint8_t  last_spike;     // 1 if last sample was classified as a spike
} KalmanFilter_t;

/* ================================================================
 *              FIXED-PARAMETER KALMAN FILTER STRUCT
 *  Same equations, but Q and R never change – the baseline that we
 *  compare against to prove the adaptive version is actually doing
 *  something useful.
 * ================================================================ */
typedef struct {
    float    x;
    float    P;
    float    Q;          // constant process noise
    float    R;          // constant measurement noise
    uint32_t last_time;
    uint8_t  initialized;
    float    last_K;
} FixedKalmanFilter_t;

/* ================================================================
 *                    GLOBAL STATE
 * ================================================================ */
KalmanFilter_t       tempFilter;      // adaptive
FixedKalmanFilter_t  fixedFilter;     // fixed-parameter baseline

float rawTemperature      = 25.0f;
float rawHumidity         = 50.0f;
float filteredTemperature = 25.0f;
float fixedTemperature    = 25.0f;

/* ================================================================
 *                I²C BIT-BANG  (unchanged – condensed)
 * ================================================================ */
void i2c_sda_low(void)    { pinMode(PIN_SDA, OUTPUT); digitalWrite(PIN_SDA, LOW); }
void i2c_sda_release(void){ pinMode(PIN_SDA, INPUT); }
void i2c_scl_low(void)    { pinMode(PIN_SCL, OUTPUT); digitalWrite(PIN_SCL, LOW); }
void i2c_scl_release(void){ pinMode(PIN_SCL, INPUT); }
uint8_t i2c_sda_read(void){ pinMode(PIN_SDA, INPUT); return digitalRead(PIN_SDA); }
void i2c_delay(void)      { delayMicroseconds(I2C_DELAY_US); }

void i2c_init(void){ i2c_sda_release(); i2c_scl_release(); i2c_delay(); }
void i2c_start(void){
    i2c_sda_release(); i2c_scl_release(); i2c_delay();
    i2c_sda_low();     i2c_delay();
    i2c_scl_low();     i2c_delay();
}
void i2c_stop(void){
    i2c_sda_low();     i2c_scl_release(); i2c_delay();
    i2c_sda_release(); i2c_delay();
}
uint8_t i2c_write_byte(uint8_t data){
    for (int i = 7; i >= 0; i--) {
        if (data & (1 << i)) i2c_sda_release(); else i2c_sda_low();
        i2c_delay(); i2c_scl_release(); i2c_delay(); i2c_scl_low();
    }
    i2c_sda_release(); i2c_delay();
    i2c_scl_release(); i2c_delay();
    uint8_t ack = i2c_sda_read();
    i2c_scl_low(); i2c_delay();
    return ack;
}

/* ================================================================
 *                    LCD2004 I²C DRIVER (condensed)
 * ================================================================ */
#if FEATURE_LCD
#define LCD_BACKLIGHT 0x08
#define LCD_ENABLE    0x04
#define LCD_RS        0x01

bool lcd_write_i2c(uint8_t data){
    i2c_start();
    i2c_write_byte(LCD_ADDR << 1);
    i2c_write_byte(data);
    i2c_stop();
    return true;
}
void lcd_write_nibble(uint8_t nibble, uint8_t rs){
    uint8_t d = (nibble & 0xF0) | LCD_BACKLIGHT | (rs ? LCD_RS : 0);
    lcd_write_i2c(d | LCD_ENABLE); delayMicroseconds(1);
    lcd_write_i2c(d & ~LCD_ENABLE); delayMicroseconds(50);
}
void lcd_write_byte(uint8_t data, uint8_t rs){
    lcd_write_nibble(data & 0xF0, rs);
    lcd_write_nibble((data << 4) & 0xF0, rs);
}
void lcd_command(uint8_t cmd){ lcd_write_byte(cmd, 0); if (cmd < 4) delay(2); }
void lcd_data(uint8_t d){ lcd_write_byte(d, 1); }
bool lcd_init(void){
    delay(50);
    lcd_write_nibble(0x30, 0); delay(5);
    lcd_write_nibble(0x30, 0); delay(1);
    lcd_write_nibble(0x30, 0); delay(1);
    lcd_write_nibble(0x20, 0); delay(1);
    lcd_command(0x28); lcd_command(0x0C); lcd_command(0x06); lcd_command(0x01);
    delay(2);
    return true;
}
void lcd_set_cursor(uint8_t row, uint8_t col){
    const uint8_t off[] = {0x00, 0x40, 0x14, 0x54};
    lcd_command(0x80 | (off[row] + col));
}
void lcd_print(const char *s){ while (*s) lcd_data(*s++); }
void lcd_clear(void){ lcd_command(0x01); delay(2); }
#else
bool lcd_init(void){ return true; }
void lcd_set_cursor(uint8_t, uint8_t){}
void lcd_print(const char *){}
void lcd_clear(void){}
#endif

/* ================================================================
 *                    DHT22 DRIVER (used only in live mode)
 * ================================================================ */
#if FEATURE_DHT
bool dht22_read(float *temperature, float *humidity){
    uint8_t data[5] = {0};
    uint8_t bit_idx = 7, byte_idx = 0;

    pinMode(PIN_DHT, OUTPUT);
    digitalWrite(PIN_DHT, LOW);  delay(18);
    digitalWrite(PIN_DHT, HIGH); delayMicroseconds(30);
    pinMode(PIN_DHT, INPUT_PULLUP);
    delayMicroseconds(10);

    uint32_t timeout;
    timeout = micros() + 100;
    while (digitalRead(PIN_DHT) == HIGH) if (micros() > timeout) return false;
    timeout = micros() + 100;
    while (digitalRead(PIN_DHT) == LOW)  if (micros() > timeout) return false;
    timeout = micros() + 100;
    while (digitalRead(PIN_DHT) == HIGH) if (micros() > timeout) return false;

    for (int i = 0; i < 40; i++) {
        timeout = micros() + 70;
        while (digitalRead(PIN_DHT) == LOW)  if (micros() > timeout) return false;
        uint32_t ts = micros();
        timeout = ts + 100;
        while (digitalRead(PIN_DHT) == HIGH) if (micros() > timeout) return false;
        if ((micros() - ts) > 40) data[byte_idx] |= (1 << bit_idx);
        if (bit_idx == 0) { bit_idx = 7; byte_idx++; } else bit_idx--;
    }
    if (((data[0]+data[1]+data[2]+data[3]) & 0xFF) != data[4]) return false;
    *humidity = ((data[0]<<8)|data[1]) / 10.0f;
    int16_t t = ((data[2]&0x7F)<<8) | data[3];
    if (data[2] & 0x80) t = -t;
    *temperature = t / 10.0f;
    return true;
}
#endif

/* ================================================================
 *                ADAPTIVE KALMAN FILTER IMPLEMENTATION
 *      (Instrumented version – stores K, rate, spike flag.)
 * ================================================================ */
void Kalman_Init(KalmanFilter_t *kf, float sensor_sigma, float max_phys_rate){
    kf->x             = 0.0f;
    kf->P             = 1.0f;
    kf->Q_base        = 1e-4f;
    kf->Q_max         = 1.0f;
    kf->R_trust       = sensor_sigma * sensor_sigma;
    kf->R_reject      = 100.0f * kf->R_trust;
    kf->Q             = kf->Q_base;
    kf->R             = kf->R_trust;
    kf->max_phys_rate = max_phys_rate;
    kf->last_time     = 0;
    kf->initialized   = 0;
    kf->last_K        = 0.0f;
    kf->last_rate     = 0.0f;
    kf->last_innov    = 0.0f;
    kf->last_spike    = 0;
}

static void Kalman_Tune(KalmanFilter_t *kf, float measurement, float dt_s){
    if (dt_s < 1e-3f) dt_s = 1e-3f;
    float innovation    = fabsf(measurement - kf->x);
    float apparent_rate = innovation / dt_s;
    kf->last_rate = apparent_rate;

    if (apparent_rate > kf->max_phys_rate) {
        kf->R = kf->R_reject;
        kf->Q = kf->Q_base;
        kf->last_spike = 1;
    } else {
        kf->R = kf->R_trust;
        float q = kf->Q_base + (apparent_rate * apparent_rate) * dt_s;
        if (q > kf->Q_max) q = kf->Q_max;
        kf->Q = q;
        kf->last_spike = 0;
    }
}

float Kalman_Update(KalmanFilter_t *kf, float measurement){
    uint32_t now = millis();
    if (!kf->initialized) {
        kf->x = measurement;
        kf->P = 1.0f;
        kf->last_time = now;
        kf->initialized = 1;
        kf->last_K = 0.0f;
        return kf->x;
    }
    float dt_s = (float)(now - kf->last_time) / 1000.0f;
    kf->last_time = now;

    Kalman_Tune(kf, measurement, dt_s);

    float P_pred = kf->P + kf->Q;
    float K      = P_pred / (P_pred + kf->R);
    float innov  = measurement - kf->x;
    kf->x        = kf->x + K * innov;
    kf->P        = (1.0f - K) * P_pred;

    kf->last_K     = K;
    kf->last_innov = innov;
    return kf->x;
}

/* ================================================================
 *                FIXED-PARAMETER KALMAN FILTER
 *  Baseline reference.  Same predict/update equations but Q and R
 *  are never touched after initialisation.
 * ================================================================ */
void FixedKF_Init(FixedKalmanFilter_t *kf, float Q, float R){
    kf->x = 0.0f;
    kf->P = 1.0f;
    kf->Q = Q;
    kf->R = R;
    kf->last_time = 0;
    kf->initialized = 0;
    kf->last_K = 0.0f;
}

float FixedKF_Update(FixedKalmanFilter_t *kf, float z){
    uint32_t now = millis();
    if (!kf->initialized) {
        kf->x = z;
        kf->P = 1.0f;
        kf->last_time = now;
        kf->initialized = 1;
        return kf->x;
    }
    kf->last_time = now;
    float P_pred = kf->P + kf->Q;
    float K      = P_pred / (P_pred + kf->R);
    kf->x = kf->x + K * (z - kf->x);
    kf->P = (1.0f - K) * P_pred;
    kf->last_K = K;
    return kf->x;
}

/* ================================================================
 *      SYNTHETIC SCENARIO GENERATOR (ground truth + noise + spike)
 *
 *  Timeline:
 *    Time (s)   Scenario              True T (°C)
 *    0 – 60     Steady_25C            25.0
 *    60 – 160   SlowRamp_25to35       25 → 35  (0.10 °C/s)
 *    160 – 220  Steady_35C            35.0
 *    220 – 240  FastRamp_35to45       35 → 45  (0.50 °C/s)
 *    240 – 260  Steady_45C            45.0
 *    t = 260    Spike (1 sample)      measurement forced to 55 °C
 *    260 – 320  PostSpike_45C         45.0
 *    320 – 360  CoolRamp_45to30       45 → 30  (-0.375 °C/s)
 *    360 – 420  Steady_30C            30.0
 * ================================================================ */
float scenario_truth(float t){
    if      (t <  60.0f) return 25.0f;
    else if (t < 160.0f) return 25.0f + (t -  60.0f) * (10.0f / 100.0f);
    else if (t < 220.0f) return 35.0f;
    else if (t < 240.0f) return 35.0f + (t - 220.0f) * (10.0f /  20.0f);
    else if (t < 260.0f) return 45.0f;
    else if (t < 320.0f) return 45.0f;
    else if (t < 360.0f) return 45.0f - (t - 320.0f) * (15.0f /  40.0f);
    else                  return 30.0f;
}

uint8_t scenario_id(float t){
    if      (t <  60.0f) return 1;
    else if (t < 160.0f) return 2;
    else if (t < 220.0f) return 3;
    else if (t < 240.0f) return 4;
    else if (t < 260.0f) return 5;
    else if (t < 320.0f) return 6;
    else if (t < 360.0f) return 7;
    else                  return 8;
}

const char* scenario_name(uint8_t id){
    switch (id) {
        case 1: return "Steady_25C";
        case 2: return "SlowRamp_25to35";
        case 3: return "Steady_35C";
        case 4: return "FastRamp_35to45";
        case 5: return "Steady_45C";
        case 6: return "PostSpike_45C";
        case 7: return "CoolRamp_45to30";
        case 8: return "Steady_30C";
        default: return "End";
    }
}

/* Box-Muller Gaussian noise.  Uses Arduino's random() (32-bit LCG).
 * With a fixed seed this generator is deterministic, so the noise
 * sequence is identical for both filters AND identical between runs.
 * That is exactly the "same input measurement sequence" controlled
 * variable required by the experimental design.                     */
float gauss_noise(float sigma){
    /* random(min, max) returns int in [min, max).  Avoid u1 = 0 since
     * log(0) is -inf.                                                 */
    float u1 = (float)random(1, 10001) / 10000.0f;
    float u2 = (float)random(0, 10001) / 10000.0f;
    float z  = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return z * sigma;
}

/* ================================================================
 *           PER-SCENARIO STATISTICS (online RMSE accumulator)
 * ================================================================ */
typedef struct {
    uint32_t n;
    double   sse_raw;    // sum of squared error (measurement vs truth)
    double   sse_fixed;  // sum of squared error (fixed filter vs truth)
    double   sse_adapt;  // sum of squared error (adaptive vs truth)
} ScenarioStats_t;

ScenarioStats_t stats[9];      // indices 1..8 used

void stats_init(void){
    for (int i = 0; i < 9; i++) {
        stats[i].n         = 0;
        stats[i].sse_raw   = 0.0;
        stats[i].sse_fixed = 0.0;
        stats[i].sse_adapt = 0.0;
    }
}

void stats_accumulate(uint8_t id, float truth, float raw,
                      float fixed, float adapt){
    if (id < 1 || id > 8) return;
    double er = (double)(raw   - truth);
    double ef = (double)(fixed - truth);
    double ea = (double)(adapt - truth);
    stats[id].n++;
    stats[id].sse_raw   += er * er;
    stats[id].sse_fixed += ef * ef;
    stats[id].sse_adapt += ea * ea;
}

void stats_report(void){
    Serial.println();
    Serial.println(F("# ===== RMSE PER SCENARIO (°C) ====="));
    Serial.println(F("# id  scenario           N      RMSE_raw   RMSE_fixed  RMSE_adapt"));
    double tot_raw=0, tot_fix=0, tot_adp=0;
    uint32_t tot_n = 0;
    for (int i = 1; i <= 8; i++) {
        if (stats[i].n == 0) continue;
        float r = sqrtf((float)(stats[i].sse_raw   / stats[i].n));
        float f = sqrtf((float)(stats[i].sse_fixed / stats[i].n));
        float a = sqrtf((float)(stats[i].sse_adapt / stats[i].n));
        Serial.print(F("# "));
        Serial.print(i);          Serial.print(F("   "));
        Serial.print(scenario_name(i));
        for (int p = strlen(scenario_name(i)); p < 18; p++) Serial.print(' ');
        Serial.print(stats[i].n); Serial.print(F("     "));
        Serial.print(r, 4);       Serial.print(F("      "));
        Serial.print(f, 4);       Serial.print(F("      "));
        Serial.println(a, 4);
        tot_raw += stats[i].sse_raw;
        tot_fix += stats[i].sse_fixed;
        tot_adp += stats[i].sse_adapt;
        tot_n   += stats[i].n;
    }
    if (tot_n > 0) {
        Serial.print(F("# OVERALL  N="));
        Serial.print(tot_n);
        Serial.print(F("   RMSE_raw="));   Serial.print(sqrtf(tot_raw/tot_n), 4);
        Serial.print(F("   RMSE_fixed=")); Serial.print(sqrtf(tot_fix/tot_n), 4);
        Serial.print(F("   RMSE_adapt=")); Serial.println(sqrtf(tot_adp/tot_n), 4);
    }
    Serial.println(F("# =================================="));
}

/* ================================================================
 *                    LCD STATUS DISPLAY
 * ================================================================ */
void update_display_testbench(float t, uint8_t sid,
                              float truth, float fixed, float adapt){
#if FEATURE_LCD
    char buf[21];
    lcd_set_cursor(0, 0); lcd_print("KF Testbench  v1.0  ");
    lcd_set_cursor(1, 0);
    snprintf(buf, sizeof(buf), "t=%4d s  Scn=%d      ", (int)t, sid);
    lcd_print(buf);
    lcd_set_cursor(2, 0);
    snprintf(buf, sizeof(buf), "True=%2d.%d  Fix=%2d.%d ",
             (int)truth, abs((int)(truth*10)%10),
             (int)fixed, abs((int)(fixed*10)%10));
    lcd_print(buf);
    lcd_set_cursor(3, 0);
    snprintf(buf, sizeof(buf), "Adapt=%2d.%d K=%d.%02d ",
             (int)adapt, abs((int)(adapt*10)%10),
             (int)tempFilter.last_K,
             abs((int)(tempFilter.last_K*100)%100));
    lcd_print(buf);
#else
    (void)t;(void)sid;(void)truth;(void)fixed;(void)adapt;
#endif
}

void update_display_live(float raw_t, float filt_t, float hum){
#if FEATURE_LCD
    char buf[21];
    lcd_set_cursor(0, 0); lcd_print("=Kalman Live Demo= ");
    lcd_set_cursor(1, 0);
    snprintf(buf, sizeof(buf), "Raw:  %d.%d C       ",
             (int)raw_t, abs((int)(raw_t*10)%10));
    lcd_print(buf);
    lcd_set_cursor(2, 0);
    snprintf(buf, sizeof(buf), "Filt: %d.%02d C      ",
             (int)filt_t, abs((int)(filt_t*100)%100));
    lcd_print(buf);
    lcd_set_cursor(3, 0);
    snprintf(buf, sizeof(buf), "Humidity: %d.%d %%   ",
             (int)hum, abs((int)(hum*10)%10));
    lcd_print(buf);
#else
    (void)raw_t;(void)filt_t;(void)hum;
#endif
}

/* ================================================================
 *                    SETUP
 * ================================================================ */
void setup(){
    Serial.begin(115200);
    delay(500);

    Serial.println();
    Serial.println(F("# ============================================"));
#if FEATURE_TESTBENCH
    Serial.println(F("#   Kalman Filter SCIENTIFIC TESTBENCH"));
#else
    Serial.println(F("#   Kalman Filter LIVE DEMO (DHT22)"));
#endif
    Serial.println(F("# ============================================"));

    /* --- Print full experimental configuration as comment header --- */
    Serial.print(F("# build_time = ")); Serial.print(__DATE__);
    Serial.print(F(" ")); Serial.println(__TIME__);
#if FEATURE_TESTBENCH
    Serial.print(F("# random_seed     = ")); Serial.println(TEST_RANDOM_SEED);
    Serial.print(F("# sample_period_s = ")); Serial.println(TEST_SAMPLE_MS / 1000.0f, 3);
    Serial.print(F("# duration_s      = ")); Serial.println(TEST_DURATION_S, 1);
    Serial.print(F("# noise_sigma_C   = ")); Serial.println(TEST_NOISE_SIGMA, 3);
    Serial.print(F("# spike_time_s    = ")); Serial.println(TEST_SPIKE_TIME_S, 1);
    Serial.print(F("# spike_value_C   = ")); Serial.println(TEST_SPIKE_VALUE, 2);
    Serial.print(F("# fixed_Q         = ")); Serial.println(FIXED_KF_Q, 5);
    Serial.print(F("# fixed_R         = ")); Serial.println(FIXED_KF_R, 5);
    Serial.print(F("# adapt_sigma     = ")); Serial.println(ADAPT_KF_SIGMA, 3);
    Serial.print(F("# adapt_max_rate  = ")); Serial.println(ADAPT_KF_MAXRATE, 3);
#endif

    /* --- I²C + LCD --- */
    i2c_init(); delay(100);
#if FEATURE_LCD
    lcd_init();
    lcd_clear();
    lcd_set_cursor(0, 0); lcd_print("KF Testbench");
    lcd_set_cursor(1, 0); lcd_print("Starting...");
#endif

    /* --- Filter init --- */
    Kalman_Init  (&tempFilter,  ADAPT_KF_SIGMA, ADAPT_KF_MAXRATE);
    FixedKF_Init (&fixedFilter, FIXED_KF_Q,     FIXED_KF_R);
    stats_init();

#if FEATURE_TESTBENCH
    /* Deterministic PRNG → reproducible noise sequence. */
    randomSeed(TEST_RANDOM_SEED);

    /* CSV header.  Every line that does NOT start with '#' is data,
     * so a parser can simply skip '#' lines (pandas: comment='#').   */
    Serial.println();
    Serial.println(F("t_s,scenario,truth,measurement,fixed_x,adapt_x,"
                     "adapt_Q,adapt_R,adapt_K,adapt_P,adapt_rate,spike_flag"));
#else
    Serial.println();
    Serial.println(F("# Time(s)  Raw(C)   Filtered(C)"));
#endif
    delay(500);
#if FEATURE_LCD
    lcd_clear();
#endif
}

/* ================================================================
 *                    TESTBENCH LOOP
 * ================================================================ */
#if FEATURE_TESTBENCH
void loop_testbench(){
    static unsigned long start_time  = millis();
    static unsigned long last_sample = 0;
    static unsigned long last_lcd    = 0;
    static bool          spike_done  = false;
    static bool          finished    = false;

    unsigned long now = millis();
    if (finished) return;

    if (now - last_sample < (unsigned long)TEST_SAMPLE_MS) return;
    last_sample = now;

    float t = (now - start_time) / 1000.0f;

    /* End-of-run: print RMSE summary and stop generating data. */
    if (t > TEST_DURATION_S) {
        stats_report();
        Serial.println(F("# === Experiment complete ==="));
#if FEATURE_LCD
        lcd_clear();
        lcd_set_cursor(0, 0); lcd_print("Experiment DONE");
        lcd_set_cursor(1, 0); lcd_print("See serial CSV.");
#endif
        finished = true;
        return;
    }

    /* ---- Ground truth + scenario ID ---- */
    float   truth = scenario_truth(t);
    uint8_t sid   = scenario_id(t);

    /* ---- Synthetic measurement: truth + Gaussian noise ---- */
    float measurement = truth + gauss_noise(TEST_NOISE_SIGMA);

    /* ---- Spike injection: single sample at TEST_SPIKE_TIME_S ---- */
    uint8_t spike_flag = 0;
    if (!spike_done && t >= TEST_SPIKE_TIME_S) {
        measurement = TEST_SPIKE_VALUE;
        spike_flag  = 1;
        spike_done  = true;
    }

    /* ---- Run BOTH filters on the SAME measurement ---- */
    float fx = FixedKF_Update(&fixedFilter, measurement);
    float ax = Kalman_Update (&tempFilter,  measurement);

    /* ---- Accumulate per-scenario RMSE ---- */
    stats_accumulate(sid, truth, measurement, fx, ax);

    /* ---- CSV log (one row per sample) ---- */
    Serial.print(t, 3);              Serial.print(',');
    Serial.print(sid);               Serial.print(',');
    Serial.print(truth, 3);          Serial.print(',');
    Serial.print(measurement, 3);    Serial.print(',');
    Serial.print(fx, 3);             Serial.print(',');
    Serial.print(ax, 3);             Serial.print(',');
    Serial.print(tempFilter.Q, 5);   Serial.print(',');
    Serial.print(tempFilter.R, 5);   Serial.print(',');
    Serial.print(tempFilter.last_K, 5); Serial.print(',');
    Serial.print(tempFilter.P, 5);   Serial.print(',');
    Serial.print(tempFilter.last_rate, 4); Serial.print(',');
    Serial.println(spike_flag);

    /* ---- LCD refresh (every ~500 ms to keep bandwidth available) ---- */
    if (now - last_lcd >= 500) {
        last_lcd = now;
        update_display_testbench(t, sid, truth, fx, ax);
    }
}
#endif  /* FEATURE_TESTBENCH */

/* ================================================================
 *                    LIVE-DEMO LOOP (DHT22)
 * ================================================================ */
#if !FEATURE_TESTBENCH
void loop_live(){
    static unsigned long start_time  = millis();
    static unsigned long last_read   = 0;
    static unsigned long last_lcd    = 0;
    unsigned long now = millis();

    if (now - last_read >= 2000) {
        last_read = now;
        float t = (now - start_time) / 1000.0f;
        float new_t, new_h;
        if (dht22_read(&new_t, &new_h)) {
            rawTemperature      = new_t;
            rawHumidity         = new_h;
            filteredTemperature = Kalman_Update(&tempFilter, rawTemperature);
            fixedTemperature    = FixedKF_Update(&fixedFilter, rawTemperature);
            Serial.print(t, 1);   Serial.print(F("  "));
            Serial.print(rawTemperature, 2);      Serial.print(F("  "));
            Serial.print(fixedTemperature, 2);    Serial.print(F("  "));
            Serial.println(filteredTemperature, 2);
        }
    }
    if (now - last_lcd >= 500) {
        last_lcd = now;
        update_display_live(rawTemperature, filteredTemperature, rawHumidity);
    }
}
#endif

/* ================================================================
 *                    MAIN LOOP DISPATCH
 * ================================================================ */
void loop(){    
#if FEATURE_TESTBENCH
    loop_testbench();
#else
    loop_live();
#endif
}