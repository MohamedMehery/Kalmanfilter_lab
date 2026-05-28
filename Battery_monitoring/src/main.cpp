#include <Arduino.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ================================================================
 *  FILE: main.cpp
 *  PROJECT: TinyEP-Lite v0.4
 *
 *  v0.4 — Eliminates ALL I²C from the sample-output critical path
 *
 *  ROOT-CAUSE OF v0.3 CORRUPTION:
 *    In Wokwi, delayMicroseconds() advances the *simulated* clock.
 *    Bit-bang I²C transactions consume thousands of simulated µs,
 *    which disrupts the simulated UART byte timing even if
 *    Serial.flush() was called beforehand.
 *
 *  v0.4 FIXES:
 *    FIX-A  Synthetic HH:MM:SS timestamps computed from sample_idx.
 *           ZERO I²C for timekeeping during the experiment.
 *           RTC is only read once at startup to prove it works.
 *
 *    FIX-B  Entire CSV row formatted into a single char[] buffer
 *           using dtostrf() for floats and snprintf() for integers,
 *           then printed with ONE Serial.println(buffer) call.
 *           No interleaved function calls that could block.
 *
 *    FIX-C  LCD updates completely isolated from serial output.
 *           Only done every 60 seconds, AFTER Serial.flush() +
 *           delay(5ms) safety gap, in a phase where no serial
 *           printing occurs.
 *
 *    FIX-D  RTC verified once at boot, then never touched again
 *           during the experiment.  g_hms_cache is updated from
 *           the synthetic clock, not from I²C.
 * ================================================================ */

/* ================================================================
 *                    DEBUG & FEATURE TOGGLES
 * ================================================================ */
#define DEBUG_ENABLE              1
#define DEBUG_VERBOSE             0

#define FEATURE_LCD               1
#define FEATURE_RTC               1
#define FEATURE_DHT               1
#define FEATURE_KALMAN            1

#define FEATURE_TESTBENCH         0
#define FEATURE_ENERGY_TESTBENCH  1

#if (FEATURE_TESTBENCH && FEATURE_ENERGY_TESTBENCH)
#error "Set only one of FEATURE_TESTBENCH / FEATURE_ENERGY_TESTBENCH"
#endif

/* ================================================================
 *                TEMPERATURE TESTBENCH CONFIG (legacy)
 * ================================================================ */
#define TEST_SAMPLE_MS            250
#define TEST_DURATION_S           420.0f
#define TEST_RANDOM_SEED          12345UL
#define TEST_NOISE_SIGMA          0.5f
#define TEST_SPIKE_TIME_S         260.0f
#define TEST_SPIKE_VALUE          55.0f
#define FIXED_KF_Q                0.01f
#define FIXED_KF_R                0.25f
#define ADAPT_KF_SIGMA            0.5f
#define ADAPT_KF_MAXRATE          1.0f

/* ================================================================
 *        ENERGY TESTBENCH CONFIG  (TinyEP-Lite v0.4)
 * ================================================================ */
#define ENERGY_SAMPLE_MS          100
#define ENERGY_DURATION_S         300.0f
#define ENERGY_RANDOM_SEED        4242UL

#define SYS_VOLTAGE               3.3f
#define ADC_BITS                  12
#define ADC_MAX                   4095.0f
#define ADC_VREF                  3.3f
#define SHUNT_R                   1.0f
#define AMP_GAIN                  10.0f

#define ADC_NOISE_SIGMA_V         0.003f
#define TEMP_DRIFT_COEFF          0.0015f
#define ADC_NONLINEAR_COEFF       0.015f

#define FUEL_PERIOD_S             5.0f
#define FEEDBACK_LR               0.02f

#define ENERGY_KF_Q               1e-5f
#define ENERGY_KF_R               1e-4f
#define ENERGY_KF_P_MAX           1.0f

#define ENERGY_ADAPT_SIGMA        0.015f
#define ENERGY_ADAPT_MAXRATE      5.0f

#define ENV_MODE_CONST_25         1
#define ENV_MODE_RAMP_25_60       2
#define ENV_MODE_HARSH_0_80       3
#define ENV_MODE                  ENV_MODE_RAMP_25_60

/* LCD refresh interval during experiment (ms).
 * Set high to minimise I²C overhead in the Wokwi simulator. */
#define LCD_REFRESH_INTERVAL_MS   60000UL

/* ================================================================
 *             NUMERICAL SAFETY
 * ================================================================ */
#define SENTINEL_NAN    -9.99f
#define SENTINEL_POSINF  9.99f
#define SENTINEL_NEGINF -9.99f

static inline float safe_float(float v) {
    if (isnan(v))           return SENTINEL_NAN;
    if (isinf(v) && v > 0) return SENTINEL_POSINF;
    if (isinf(v))           return SENTINEL_NEGINF;
    return v;
}
static inline float safe_d(double v) {
    if (isnan(v))           return SENTINEL_NAN;
    if (isinf(v) && v > 0) return SENTINEL_POSINF;
    if (isinf(v))           return SENTINEL_NEGINF;
    if (v >  1.0e9)  return  1.0e9f;
    if (v < -1.0e9)  return -1.0e9f;
    return (float)v;
}

/* ================================================================
 *    SINGLE-BUFFER CSV ROW BUILDER
 *
 *    All CSV fields are appended to one char[] buffer using
 *    dtostrf() for floats (guaranteed available on all Arduino
 *    cores, unlike snprintf %f).  The completed buffer is then
 *    printed with a single Serial.println(buf) — NO interleaved
 *    function calls that could block or corrupt UART.
 * ================================================================ */
#define CSV_BUF_SIZE 256

static char  g_csv[CSV_BUF_SIZE];
static int   g_csv_pos;

static void csv_reset(void) {
    g_csv_pos = 0;
    g_csv[0]  = '\0';
}

static void csv_append_ch(char c) {
    if (g_csv_pos < CSV_BUF_SIZE - 1) {
        g_csv[g_csv_pos++] = c;
        g_csv[g_csv_pos]   = '\0';
    }
}

static void csv_append_str(const char *s) {
    while (*s && g_csv_pos < CSV_BUF_SIZE - 1) {
        g_csv[g_csv_pos++] = *s++;
    }
    g_csv[g_csv_pos] = '\0';
}

static void csv_append_float(float val, int decimals) {
    char tmp[16];
    dtostrf(val, 1, decimals, tmp);
    csv_append_str(tmp);
}

static void csv_append_uint(unsigned int val) {
    char tmp[12];
    snprintf(tmp, sizeof(tmp), "%u", val);
    csv_append_str(tmp);
}

/* Append comma-separated float field */
static void csv_field_f(float val, int decimals) {
    csv_append_float(val, decimals);
    csv_append_ch(',');
}

/* Append comma-separated unsigned field */
static void csv_field_u(unsigned int val) {
    csv_append_uint(val);
    csv_append_ch(',');
}

/* Append comma-separated string field */
static void csv_field_s(const char *s) {
    csv_append_str(s);
    csv_append_ch(',');
}

/* ================================================================
 *    SYNTHETIC TIMESTAMP
 *
 *    FIX-A: Compute HH:MM:SS from the deterministic sample index.
 *    No I²C needed.  The RTC is only read at boot to verify it
 *    works, then never touched again during the experiment.
 * ================================================================ */
static void format_synthetic_hms(char *buf, size_t n, float t_seconds) {
    uint32_t total = (uint32_t)t_seconds;
    unsigned hh = total / 3600U;
    unsigned mm = (total % 3600U) / 60U;
    unsigned ss = total % 60U;
    snprintf(buf, n, "%02u:%02u:%02u", hh, mm, ss);
}

/* ================================================================
 *                    DEBUG MACROS
 * ================================================================ */
#if DEBUG_ENABLE
  #define DEBUG_PRINT(...)   Serial.print(__VA_ARGS__)
  #define DEBUG_PRINTLN(...) Serial.println(__VA_ARGS__)
#else
  #define DEBUG_PRINT(...)
  #define DEBUG_PRINTLN(...)
#endif

/* ================================================================
 *                    PIN / I²C DEFINITIONS
 * ================================================================ */
#define PIN_SDA      PB7
#define PIN_SCL      PB6
#define PIN_DHT      PA0
#define LCD_ADDR     0x27
#define RTC_ADDR     0x68
#define I2C_DELAY_US 500

/* ================================================================
 *              KALMAN FILTER STRUCTS
 * ================================================================ */
typedef struct {
    float    x, P, Q, R;
    float    Q_base, Q_max;
    float    R_trust, R_reject;
    float    max_phys_rate;
    uint32_t last_time;
    uint8_t  initialized;
    float    last_K, last_rate, last_innov;
    uint8_t  last_spike;
} KalmanFilter_t;

typedef struct {
    float    x, P, Q, R, P_max;
    uint32_t last_time;
    uint8_t  initialized;
    float    last_K;
} FixedKalmanFilter_t;

/* ================================================================
 *                    GLOBAL STATE
 * ================================================================ */
KalmanFilter_t       tempFilter;
FixedKalmanFilter_t  fixedFilter;
float rawTemperature      = 25.0f;
float rawHumidity         = 50.0f;
float filteredTemperature = 25.0f;
float fixedTemperature    = 25.0f;

/* ================================================================
 *                I²C BIT-BANG
 * ================================================================ */
void i2c_sda_low(void)    { pinMode(PIN_SDA,OUTPUT); digitalWrite(PIN_SDA,LOW); }
void i2c_sda_release(void){ pinMode(PIN_SDA,INPUT); }
void i2c_scl_low(void)    { pinMode(PIN_SCL,OUTPUT); digitalWrite(PIN_SCL,LOW); }
void i2c_scl_release(void){ pinMode(PIN_SCL,INPUT); }
uint8_t i2c_sda_read(void){ pinMode(PIN_SDA,INPUT); return digitalRead(PIN_SDA); }
void i2c_delay(void)      { delayMicroseconds(I2C_DELAY_US); }

void i2c_init(void){ i2c_sda_release(); i2c_scl_release(); i2c_delay(); }
void i2c_start(void){
    i2c_sda_release(); i2c_scl_release(); i2c_delay();
    i2c_sda_low(); i2c_delay();
    i2c_scl_low(); i2c_delay();
}
void i2c_stop(void){
    i2c_sda_low(); i2c_scl_release(); i2c_delay();
    i2c_sda_release(); i2c_delay();
}
uint8_t i2c_write_byte(uint8_t data){
    for(int i=7;i>=0;i--){
        if(data&(1<<i)) i2c_sda_release(); else i2c_sda_low();
        i2c_delay(); i2c_scl_release(); i2c_delay(); i2c_scl_low();
    }
    i2c_sda_release(); i2c_delay();
    i2c_scl_release(); i2c_delay();
    uint8_t ack=i2c_sda_read();
    i2c_scl_low(); i2c_delay();
    return ack;
}
uint8_t i2c_read_byte(uint8_t send_nack){
    uint8_t data=0;
    i2c_sda_release();
    for(int i=7;i>=0;i--){
        i2c_scl_release(); i2c_delay();
        if(i2c_sda_read()) data|=(1<<i);
        i2c_scl_low(); i2c_delay();
    }
    if(send_nack) i2c_sda_release(); else i2c_sda_low();
    i2c_delay();
    i2c_scl_release(); i2c_delay();
    i2c_scl_low();
    i2c_sda_release(); i2c_delay();
    return data;
}

/* ================================================================
 *                    DS1307 RTC DRIVER
 * ================================================================ */
#if FEATURE_RTC
#define RTC_REG_SECONDS  0x00
#define RTC_REG_CONTROL  0x07
#define RTC_SQW_1HZ      0x10

static inline uint8_t bcd_to_dec(uint8_t b){ return ((b>>4)*10)+(b&0x0F); }
static inline uint8_t dec_to_bcd(uint8_t d){ return ((d/10)<<4)|(d%10); }

bool rtc_write_reg(uint8_t reg, uint8_t val){
    i2c_start();
    if(i2c_write_byte(RTC_ADDR<<1)){i2c_stop();return false;}
    if(i2c_write_byte(reg))        {i2c_stop();return false;}
    if(i2c_write_byte(val))        {i2c_stop();return false;}
    i2c_stop(); return true;
}
bool rtc_read_burst(uint8_t reg, uint8_t *buf, uint8_t n){
    i2c_start();
    if(i2c_write_byte(RTC_ADDR<<1))     {i2c_stop();return false;}
    if(i2c_write_byte(reg))             {i2c_stop();return false;}
    i2c_start();
    if(i2c_write_byte((RTC_ADDR<<1)|1)){i2c_stop();return false;}
    for(uint8_t i=0;i<n;i++)
        buf[i]=i2c_read_byte(i==(n-1)?1:0);
    i2c_stop(); return true;
}
bool rtc_init(uint8_t sqw_config){
    uint8_t sec;
    if(!rtc_read_burst(RTC_REG_SECONDS,&sec,1)) return false;
    if(sec&0x80)
        if(!rtc_write_reg(RTC_REG_SECONDS,sec&0x7F)) return false;
    return rtc_write_reg(RTC_REG_CONTROL,sqw_config);
}
bool rtc_set_time(uint8_t h, uint8_t m, uint8_t s){
    i2c_start();
    if(i2c_write_byte(RTC_ADDR<<1))       {i2c_stop();return false;}
    if(i2c_write_byte(RTC_REG_SECONDS))   {i2c_stop();return false;}
    if(i2c_write_byte(dec_to_bcd(s)&0x7F)){i2c_stop();return false;}
    if(i2c_write_byte(dec_to_bcd(m)))     {i2c_stop();return false;}
    if(i2c_write_byte(dec_to_bcd(h)&0x3F)){i2c_stop();return false;}
    i2c_stop(); return true;
}
bool rtc_get_time(uint8_t *h, uint8_t *m, uint8_t *s){
    uint8_t buf[3];
    if(!rtc_read_burst(RTC_REG_SECONDS,buf,3)) return false;
    *s=bcd_to_dec(buf[0]&0x7F);
    *m=bcd_to_dec(buf[1]&0x7F);
    *h=bcd_to_dec(buf[2]&0x3F);
    return true;
}
void rtc_format_hms(char *buf, size_t n){
    uint8_t h,m,s;
    if(rtc_get_time(&h,&m,&s))
        snprintf(buf,n,"%02u:%02u:%02u",(unsigned)h,(unsigned)m,(unsigned)s);
    else
        snprintf(buf,n,"--:--:--");
}
#else
bool rtc_init(uint8_t){return false;}
bool rtc_set_time(uint8_t,uint8_t,uint8_t){return false;}
bool rtc_get_time(uint8_t*,uint8_t*,uint8_t*){return false;}
void rtc_format_hms(char *b,size_t n){if(n)snprintf(b,n,"--:--:--");}
#endif

/* ================================================================
 *                    LCD2004 I²C DRIVER
 * ================================================================ */
#if FEATURE_LCD
#define LCD_BACKLIGHT 0x08
#define LCD_ENABLE    0x04
#define LCD_RS        0x01

bool lcd_write_i2c(uint8_t data){
    i2c_start(); i2c_write_byte(LCD_ADDR<<1); i2c_write_byte(data); i2c_stop();
    return true;
}
void lcd_write_nibble(uint8_t nibble, uint8_t rs){
    uint8_t d=(nibble&0xF0)|LCD_BACKLIGHT|(rs?LCD_RS:0);
    lcd_write_i2c(d|LCD_ENABLE);  delayMicroseconds(1);
    lcd_write_i2c(d&~LCD_ENABLE); delayMicroseconds(50);
}
void lcd_write_byte(uint8_t data, uint8_t rs){
    lcd_write_nibble(data&0xF0,rs);
    lcd_write_nibble((data<<4)&0xF0,rs);
}
void lcd_command(uint8_t cmd){ lcd_write_byte(cmd,0); if(cmd<4) delay(2); }
void lcd_data(uint8_t d){ lcd_write_byte(d,1); }
bool lcd_init(void){
    delay(50);
    lcd_write_nibble(0x30,0); delay(5);
    lcd_write_nibble(0x30,0); delay(1);
    lcd_write_nibble(0x30,0); delay(1);
    lcd_write_nibble(0x20,0); delay(1);
    lcd_command(0x28); lcd_command(0x0C); lcd_command(0x06); lcd_command(0x01);
    delay(2); return true;
}
void lcd_set_cursor(uint8_t row, uint8_t col){
    const uint8_t off[]={0x00,0x40,0x14,0x54};
    lcd_command(0x80|(off[row]+col));
}
void lcd_print(const char *s){ while(*s) lcd_data(*s++); }
void lcd_clear(void){ lcd_command(0x01); delay(2); }
#else
bool lcd_init(void){return true;}
void lcd_set_cursor(uint8_t,uint8_t){}
void lcd_print(const char*){}
void lcd_clear(void){}
#endif

/* ================================================================
 *   LCD SAFE-UPDATE  (FIX-C)
 *
 *   Guarantees the UART TX buffer is fully drained AND a safety
 *   gap of 5 ms has elapsed before any I²C byte hits the bus.
 * ================================================================ */
#if FEATURE_LCD
static void lcd_safe_update(const char *l0, const char *l1,
                            const char *l2, const char *l3)
{
    Serial.flush();
    delay(5);     /* 5 ms safety gap after last UART byte */
    lcd_set_cursor(0,0); lcd_print(l0);
    lcd_set_cursor(1,0); lcd_print(l1);
    lcd_set_cursor(2,0); lcd_print(l2);
    lcd_set_cursor(3,0); lcd_print(l3);
}
#else
static void lcd_safe_update(const char*,const char*,
                            const char*,const char*){}
#endif

/* ================================================================
 *                    DHT22 DRIVER
 * ================================================================ */
#if FEATURE_DHT
bool dht22_read(float *temperature, float *humidity){
    uint8_t data[5]={0};
    uint8_t bit_idx=7,byte_idx=0;
    pinMode(PIN_DHT,OUTPUT); digitalWrite(PIN_DHT,LOW); delay(18);
    digitalWrite(PIN_DHT,HIGH); delayMicroseconds(30);
    pinMode(PIN_DHT,INPUT_PULLUP); delayMicroseconds(10);
    uint32_t timeout;
    timeout=micros()+100; while(digitalRead(PIN_DHT)==HIGH) if(micros()>timeout) return false;
    timeout=micros()+100; while(digitalRead(PIN_DHT)==LOW)  if(micros()>timeout) return false;
    timeout=micros()+100; while(digitalRead(PIN_DHT)==HIGH) if(micros()>timeout) return false;
    for(int i=0;i<40;i++){
        timeout=micros()+70; while(digitalRead(PIN_DHT)==LOW) if(micros()>timeout) return false;
        uint32_t ts=micros(); timeout=ts+100;
        while(digitalRead(PIN_DHT)==HIGH) if(micros()>timeout) return false;
        if((micros()-ts)>40) data[byte_idx]|=(1<<bit_idx);
        if(bit_idx==0){bit_idx=7;byte_idx++;}else bit_idx--;
    }
    if(((data[0]+data[1]+data[2]+data[3])&0xFF)!=data[4]) return false;
    float h=((data[0]<<8)|data[1])/10.0f;
    int16_t traw=((data[2]&0x7F)<<8)|data[3];
    if(data[2]&0x80) traw=-traw;
    float t=traw/10.0f;
    if(isnan(t)||isinf(t)||t<-40.0f||t>80.0f) return false;
    if(isnan(h)||isinf(h)||h<0.0f||h>100.0f) return false;
    *temperature=t; *humidity=h; return true;
}
#endif

/* ================================================================
 *                ADAPTIVE KALMAN FILTER
 * ================================================================ */
void Kalman_Init(KalmanFilter_t *kf, float sigma, float max_rate){
    kf->x=0; kf->P=1;
    kf->Q_base=1e-4f; kf->Q_max=1.0f;
    kf->R_trust=sigma*sigma;
    kf->R_reject=100.0f*kf->R_trust;
    kf->Q=kf->Q_base; kf->R=kf->R_trust;
    kf->max_phys_rate=max_rate;
    kf->last_time=0; kf->initialized=0;
    kf->last_K=0; kf->last_rate=0;
    kf->last_innov=0; kf->last_spike=0;
}
static void Kalman_Tune(KalmanFilter_t *kf, float meas, float dt_s){
    if(dt_s<1e-3f) dt_s=1e-3f;
    float innov=fabsf(meas-kf->x);
    float rate=innov/dt_s;
    kf->last_rate=rate;
    if(rate>kf->max_phys_rate){
        kf->R=kf->R_reject; kf->Q=kf->Q_base; kf->last_spike=1;
    } else {
        kf->R=kf->R_trust;
        float q=kf->Q_base+(rate*rate)*dt_s;
        if(q>kf->Q_max) q=kf->Q_max;
        kf->Q=q; kf->last_spike=0;
    }
}
float Kalman_Update(KalmanFilter_t *kf, float z){
    if(isnan(z)||isinf(z)) return safe_float(kf->x);
    uint32_t now=millis();
    if(!kf->initialized){
        kf->x=z; kf->P=1.0f; kf->last_time=now;
        kf->initialized=1; kf->last_K=0;
        return kf->x;
    }
    float dt_s=(float)(now-kf->last_time)/1000.0f;
    if(dt_s<1e-3f) dt_s=1e-3f;
    kf->last_time=now;
    Kalman_Tune(kf,z,dt_s);
    float P_pred=kf->P+kf->Q;
    float K=P_pred/(P_pred+kf->R);
    kf->x=kf->x+K*(z-kf->x);
    kf->P=(1.0f-K)*P_pred;
    kf->last_K=K; kf->last_innov=z-kf->x;
    if(isnan(kf->x)||isinf(kf->x)){kf->x=z;kf->P=1.0f;}
    return safe_float(kf->x);
}

/* ================================================================
 *                FIXED-PARAMETER KALMAN FILTER
 * ================================================================ */
void FixedKF_Init(FixedKalmanFilter_t *kf, float Q, float R){
    kf->x=0; kf->P=1; kf->Q=Q; kf->R=R;
    kf->P_max=10.0f*R;
    kf->last_time=0; kf->initialized=0; kf->last_K=0;
}
float FixedKF_Update(FixedKalmanFilter_t *kf, float z){
    if(isnan(z)||isinf(z)) return safe_float(kf->x);
    uint32_t now=millis();
    if(!kf->initialized){
        kf->x=z; kf->P=1.0f; kf->last_time=now; kf->initialized=1;
        return kf->x;
    }
    kf->last_time=now;
    float P_pred=kf->P+kf->Q;
    if(P_pred>kf->P_max) P_pred=kf->P_max;
    float K=P_pred/(P_pred+kf->R);
    kf->x=kf->x+K*(z-kf->x);
    kf->P=(1.0f-K)*P_pred;
    kf->last_K=K;
    if(isnan(kf->x)||isinf(kf->x)){kf->x=z;kf->P=kf->R;}
    return safe_float(kf->x);
}

/* ================================================================
 *      TEMPERATURE SCENARIO GENERATOR  (legacy)
 * ================================================================ */
float scenario_truth(float t){
    if(t< 60) return 25.0f;
    if(t<160) return 25.0f+(t-60)*(10.0f/100);
    if(t<220) return 35.0f;
    if(t<240) return 35.0f+(t-220)*(10.0f/20);
    if(t<260) return 45.0f;
    if(t<320) return 45.0f;
    if(t<360) return 45.0f-(t-320)*(15.0f/40);
    return 30.0f;
}
uint8_t scenario_id(float t){
    if(t<60) return 1; if(t<160) return 2;
    if(t<220) return 3; if(t<240) return 4;
    if(t<260) return 5; if(t<320) return 6;
    if(t<360) return 7; return 8;
}
const char* scenario_name(uint8_t id){
    switch(id){
        case 1:return"Steady_25C";     case 2:return"SlowRamp_25to35";
        case 3:return"Steady_35C";     case 4:return"FastRamp_35to45";
        case 5:return"Steady_45C";     case 6:return"PostSpike_45C";
        case 7:return"CoolRamp_45to30";default:return"Steady_30C";
    }
}
float gauss_noise(float sigma){
    float u1=(float)random(1,10001)/10000.0f;
    float u2=(float)random(0,10001)/10000.0f;
    return sigma*sqrtf(-2.0f*logf(u1))*cosf(2.0f*M_PI*u2);
}

/* ================================================================
 *           PER-SCENARIO STATISTICS  (legacy)
 * ================================================================ */
typedef struct{uint32_t n; double sse_raw,sse_fixed,sse_adapt;} ScenarioStats_t;
ScenarioStats_t stats[9];
void stats_init(void){
    for(int i=0;i<9;i++)
        stats[i].n=0,stats[i].sse_raw=stats[i].sse_fixed=stats[i].sse_adapt=0;
}
void stats_accumulate(uint8_t id,float truth,float raw,float fixed,float adapt){
    if(id<1||id>8) return;
    double er=raw-truth,ef=fixed-truth,ea=adapt-truth;
    stats[id].n++;
    stats[id].sse_raw+=er*er; stats[id].sse_fixed+=ef*ef; stats[id].sse_adapt+=ea*ea;
}
void stats_report(void){
    Serial.println();
    Serial.println(F("# ===== RMSE PER SCENARIO (C) ====="));
    double tot_raw=0,tot_fix=0,tot_adp=0; uint32_t tot_n=0;
    for(int i=1;i<=8;i++){
        if(!stats[i].n) continue;
        Serial.print(F("# ")); Serial.print(i); Serial.print(F("  "));
        Serial.print(scenario_name(i)); Serial.print(F("  "));
        Serial.print(sqrtf(stats[i].sse_raw/stats[i].n),4); Serial.print(F("  "));
        Serial.print(sqrtf(stats[i].sse_fixed/stats[i].n),4); Serial.print(F("  "));
        Serial.println(sqrtf(stats[i].sse_adapt/stats[i].n),4);
        tot_raw+=stats[i].sse_raw; tot_fix+=stats[i].sse_fixed;
        tot_adp+=stats[i].sse_adapt; tot_n+=stats[i].n;
    }
    if(tot_n>0){
        Serial.print(F("# OVERALL N=")); Serial.print(tot_n);
        Serial.print(F(" raw=")); Serial.print(sqrtf(tot_raw/tot_n),4);
        Serial.print(F(" fix=")); Serial.print(sqrtf(tot_fix/tot_n),4);
        Serial.print(F(" adp=")); Serial.println(sqrtf(tot_adp/tot_n),4);
    }
    Serial.println(F("# =================================="));
}

/* ================================================================
 *       ENERGY TESTBENCH MODULE — TinyEP-Lite v0.4
 * ================================================================ */
#if FEATURE_ENERGY_TESTBENCH

static double  E_truth=0, E_raw=0, E_kf=0, E_adapt=0, E_combo=0;
static float   affine_a=1.0f, affine_b=0.0f;
static double  fb_E_truth_win=0, fb_E_est_win=0, fb_sum_raw_win=0;
static float   fb_T_win=0, fb_last_time_s=0;
static FixedKalmanFilter_t powerFixedKF;
static KalmanFilter_t      powerAdaptKF;

float power_truth(float t){
    float p=0.030f;
    float c12=fmodf(t,12.0f);
    if(c12>2.0f&&c12<5.0f) p+=0.120f;
    if(fmodf(t,7.0f)<1.0f) p+=0.050f;
    float c20=fmodf(t,20.0f);
    if(c20>10.0f&&c20<10.25f) p+=0.350f;
    float c45=fmodf(t,45.0f);
    if(c45>30.0f&&c45<30.18f) p+=0.950f;
    p+=0.030f*sinf(2.0f*M_PI*t/60.0f);
    return p;
}

float environment_temp(float t){
#if   (ENV_MODE==ENV_MODE_CONST_25)
    return 25.0f;
#elif (ENV_MODE==ENV_MODE_RAMP_25_60)
    if(t<100) return 25.0f;
    if(t<200) return 25.0f+(t-100)*0.35f;
    return 60.0f;
#elif (ENV_MODE==ENV_MODE_HARSH_0_80)
    if(t<40) return 0.0f;
    if(t<140) return (t-40)*0.80f;
    return 80.0f;
#else
    return 25.0f;
#endif
}

static float energy_gauss(float sigma){
    float u1=(float)random(1,10001)/10000.0f;
    float u2=(float)random(0,10001)/10000.0f;
    return sigma*sqrtf(-2.0f*logf(u1))*cosf(2.0f*M_PI*u2);
}

uint16_t adc_measure_power(float power_w, float temp_c, uint8_t *clip_flag){
    float current=power_w/SYS_VOLTAGE;
    float v_shunt=current*SHUNT_R;
    float v_amp=v_shunt*AMP_GAIN;
    float temp_drift=1.0f+TEMP_DRIFT_COEFF*(temp_c-25.0f);
    float nonlinear=ADC_NONLINEAR_COEFF*v_amp*v_amp;
    float v_noisy=v_amp*temp_drift+nonlinear+energy_gauss(ADC_NOISE_SIGMA_V);
    uint8_t clipped=0;
    if(v_noisy<0) v_noisy=0;
    if(v_noisy>ADC_VREF){v_noisy=ADC_VREF;clipped=1;}
    if(clip_flag) *clip_flag=clipped;
    return (uint16_t)((v_noisy/ADC_VREF)*ADC_MAX);
}

float adc_to_power_raw(uint16_t code){
    float v_adc=((float)code/ADC_MAX)*ADC_VREF;
    return (v_adc/(AMP_GAIN*SHUNT_R))*SYS_VOLTAGE;
}

static float adaptive_power_estimate(float raw){
    return safe_float(affine_a*raw+affine_b);
}

static void sparse_feedback_update(double E_ref, double E_est,
                                   double sum_raw_dt, float T_win){
    float err=(float)(E_est-E_ref);
    float norm=(float)(sum_raw_dt*sum_raw_dt)+T_win*T_win+1e-6f;
    if(isnan(err)||isinf(err)||norm<1e-12f) return;
    float da=FEEDBACK_LR*err*(float)sum_raw_dt/norm;
    float db=FEEDBACK_LR*err*T_win/norm;
    if(!isnan(da)&&!isinf(da)) affine_a-=da;
    if(!isnan(db)&&!isinf(db)) affine_b-=db;
    if(affine_a<0.3f) affine_a=0.3f;
    if(affine_a>2.0f) affine_a=2.0f;
    if(affine_b<-0.2f) affine_b=-0.2f;
    if(affine_b> 0.2f) affine_b= 0.2f;
    if(isnan(affine_a)||isinf(affine_a)) affine_a=1.0f;
    if(isnan(affine_b)||isinf(affine_b)) affine_b=0.0f;
}

typedef struct{
    uint32_t n, fb_events, clip_events, bad_samples;
    double sse_raw, sse_kf, sse_adapt, sse_combo;
    double abs_raw, abs_kf, abs_adapt, abs_combo;
} EnergyStats_t;
static EnergyStats_t estats;

static void estats_init(void){memset(&estats,0,sizeof(estats));}

static void estats_accumulate(float truth, float raw, float kf,
                              float ad, float combo){
    if(isnan(truth)||isnan(raw)||isnan(kf)||isnan(ad)||isnan(combo)){
        estats.bad_samples++; return;
    }
    double er=raw-truth,ek=kf-truth,ea=ad-truth,ec=combo-truth;
    estats.n++;
    estats.sse_raw+=er*er; estats.sse_kf+=ek*ek;
    estats.sse_adapt+=ea*ea; estats.sse_combo+=ec*ec;
    if(truth>1e-6f){
        estats.abs_raw  +=fabs(er)/truth;
        estats.abs_kf   +=fabs(ek)/truth;
        estats.abs_adapt+=fabs(ea)/truth;
        estats.abs_combo+=fabs(ec)/truth;
    }
}

static void estats_report(void){
    Serial.flush();
    Serial.println();
    Serial.println(F("# ===== ENERGY TESTBENCH RESULTS ====="));
    if(estats.n==0){Serial.println(F("# no valid samples"));return;}
    float n=(float)estats.n;
    Serial.print(F("# valid_samples   = ")); Serial.println(estats.n);
    Serial.print(F("# bad_samples     = ")); Serial.println(estats.bad_samples);
    Serial.print(F("# feedback_events = ")); Serial.println(estats.fb_events);
    Serial.print(F("# clip_events     = ")); Serial.println(estats.clip_events);
    Serial.println();
    Serial.print(F("# RMSE_pow_raw_W  = ")); Serial.println(sqrtf(estats.sse_raw/n),6);
    Serial.print(F("# RMSE_pow_kf_W   = ")); Serial.println(sqrtf(estats.sse_kf/n),6);
    Serial.print(F("# RMSE_pow_ada_W  = ")); Serial.println(sqrtf(estats.sse_adapt/n),6);
    Serial.print(F("# RMSE_pow_cmb_W  = ")); Serial.println(sqrtf(estats.sse_combo/n),6);
    Serial.println();
    Serial.print(F("# MAPE_pow_raw_%  = ")); Serial.println(100.0f*estats.abs_raw/n,3);
    Serial.print(F("# MAPE_pow_kf_%   = ")); Serial.println(100.0f*estats.abs_kf/n,3);
    Serial.print(F("# MAPE_pow_ada_%  = ")); Serial.println(100.0f*estats.abs_adapt/n,3);
    Serial.print(F("# MAPE_pow_cmb_%  = ")); Serial.println(100.0f*estats.abs_combo/n,3);
    Serial.println();
    double Et=(E_truth>1e-9)?E_truth:1e-9;
    Serial.print(F("# E_truth_J       = ")); Serial.println(safe_d(E_truth),4);
    Serial.print(F("# E_raw_err_%     = ")); Serial.println(100.0*fabs(E_raw-E_truth)/Et,3);
    Serial.print(F("# E_kf_err_%      = ")); Serial.println(100.0*fabs(E_kf-E_truth)/Et,3);
    Serial.print(F("# E_adapt_err_%   = ")); Serial.println(100.0*fabs(E_adapt-E_truth)/Et,3);
    Serial.print(F("# E_combo_err_%   = ")); Serial.println(100.0*fabs(E_combo-E_truth)/Et,3);
    Serial.println();
    Serial.print(F("# final_affine_a  = ")); Serial.println(affine_a,5);
    Serial.print(F("# final_affine_b  = ")); Serial.println(affine_b,5);
    Serial.println(F("# ==================================="));
    Serial.flush();
}
#endif /* FEATURE_ENERGY_TESTBENCH */

/* ================================================================
 *                    SETUP
 * ================================================================ */
void setup()
{
    Serial.begin(115200);
    delay(500);

    Serial.println();
    Serial.println(F("# ============================================"));
#if FEATURE_ENERGY_TESTBENCH
    Serial.println(F("#   TinyEP-Lite v0.4  ENERGY TESTBENCH"));
#elif FEATURE_TESTBENCH
    Serial.println(F("#   Kalman Filter SCIENTIFIC TESTBENCH"));
#else
    Serial.println(F("#   Kalman Filter LIVE DEMO (DHT22)"));
#endif
    Serial.println(F("# ============================================"));
    Serial.print(F("# build_time       = "));
    Serial.print(__DATE__); Serial.print(' '); Serial.println(__TIME__);

#if FEATURE_ENERGY_TESTBENCH
    Serial.print(F("# random_seed      = ")); Serial.println(ENERGY_RANDOM_SEED);
    Serial.print(F("# energy_sample_ms = ")); Serial.println(ENERGY_SAMPLE_MS);
    Serial.print(F("# energy_duration  = ")); Serial.println(ENERGY_DURATION_S,1);
    Serial.print(F("# fuel_period_s    = ")); Serial.println(FUEL_PERIOD_S,2);
    Serial.print(F("# sys_voltage_V    = ")); Serial.println(SYS_VOLTAGE,2);
    Serial.print(F("# shunt_R_ohm      = ")); Serial.println(SHUNT_R,3);
    Serial.print(F("# amp_gain         = ")); Serial.println(AMP_GAIN,2);
    Serial.print(F("# adc_noise_sig_V  = ")); Serial.println(ADC_NOISE_SIGMA_V,4);
    Serial.print(F("# temp_drift_coef  = ")); Serial.println(TEMP_DRIFT_COEFF,5);
    Serial.print(F("# nonlinear_coef   = ")); Serial.println(ADC_NONLINEAR_COEFF,4);
    Serial.print(F("# env_mode         = ")); Serial.println(ENV_MODE);
    Serial.print(F("# feedback_lr      = ")); Serial.println(FEEDBACK_LR,4);
    Serial.print(F("# kf_Q             = ")); Serial.println(ENERGY_KF_Q,6);
    Serial.print(F("# kf_R             = ")); Serial.println(ENERGY_KF_R,6);
    Serial.print(F("# lcd_refresh_s    = ")); Serial.println(LCD_REFRESH_INTERVAL_MS/1000UL);
    Serial.println(F("# rtc_mode         = SYNTHETIC (no I2C during experiment)"));
#endif

    Serial.flush();

    /* I²C init */
    i2c_init();
    delay(50);

    /* RTC — read once to prove it works, then never again during experiment */
#if FEATURE_RTC
    if(rtc_init(RTC_SQW_1HZ)){
        rtc_set_time(0,0,0);
        char hms_check[10];
        rtc_format_hms(hms_check, sizeof(hms_check));
        Serial.print(F("# rtc_init         = OK  verified="));
        Serial.println(hms_check);
    } else {
        Serial.println(F("# rtc_init         = FAILED"));
    }
#endif

    /* LCD init */
#if FEATURE_LCD
    Serial.flush();
    lcd_init(); lcd_clear();
  #if FEATURE_ENERGY_TESTBENCH
    lcd_set_cursor(0,0); lcd_print("TinyEP-Lite v0.4");
    lcd_set_cursor(1,0); lcd_print("Starting...     ");
  #elif FEATURE_TESTBENCH
    lcd_set_cursor(0,0); lcd_print("KF Testbench    ");
    lcd_set_cursor(1,0); lcd_print("Starting...     ");
  #else
    lcd_set_cursor(0,0); lcd_print("Kalman Live Demo");
    lcd_set_cursor(1,0); lcd_print("Waiting DHT22...");
  #endif
#endif

    /* Filters */
    Kalman_Init (&tempFilter, ADAPT_KF_SIGMA, ADAPT_KF_MAXRATE);
    FixedKF_Init(&fixedFilter, FIXED_KF_Q, FIXED_KF_R);
    stats_init();

#if FEATURE_TESTBENCH
    randomSeed(TEST_RANDOM_SEED);
    Serial.println();
    Serial.println(F("t_s,rtc_hms,scenario,truth,meas,fixed_x,adapt_x,"
                     "adapt_Q,adapt_R,adapt_K,adapt_P,adapt_rate,spike"));
#endif

#if FEATURE_ENERGY_TESTBENCH
    randomSeed(ENERGY_RANDOM_SEED);
    estats_init();
    FixedKF_Init(&powerFixedKF, ENERGY_KF_Q, ENERGY_KF_R);
    powerFixedKF.P_max = ENERGY_KF_P_MAX;
    Kalman_Init(&powerAdaptKF, ENERGY_ADAPT_SIGMA, ENERGY_ADAPT_MAXRATE);
    Serial.println();
    Serial.println(F("t_s,rtc_hms,temp_c,truth_W,adc,raw_W,kf_W,adapt_W,combo_W,"
                     "E_truth_J,E_raw_J,E_kf_J,E_adapt_J,E_combo_J,"
                     "affine_a,affine_b,clip,fb"));
#endif

#if !FEATURE_TESTBENCH && !FEATURE_ENERGY_TESTBENCH
    Serial.println();
    Serial.println(F("t_s,rtc_hms,raw_C,fixed_C,adapt_C,humidity,status"));
#endif

    Serial.flush();
#if FEATURE_LCD
    delay(800);
    lcd_clear();
#endif
}

/* ================================================================
 *                    TEMPERATURE TESTBENCH LOOP (legacy)
 * ================================================================ */
#if FEATURE_TESTBENCH
void loop_testbench(){
    static bool first_tick=true;
    static unsigned long start_time=0, last_sample=0, last_lcd=0;
    static bool spike_done=false, finished=false;

    if(finished) return;
    unsigned long now=millis();
    if(first_tick){start_time=now;first_tick=false;}
    if(now-last_sample<(unsigned long)TEST_SAMPLE_MS) return;
    last_sample=now;

    float t=(now-start_time)/1000.0f;
    if(t>TEST_DURATION_S){
        stats_report();
        Serial.println(F("# === Experiment complete ==="));
        Serial.flush();
        lcd_clear(); lcd_set_cursor(0,0); lcd_print("Experiment DONE");
        finished=true; return;
    }

    float truth=scenario_truth(t);
    uint8_t sid=scenario_id(t);
    float meas=truth+gauss_noise(TEST_NOISE_SIGMA);
    uint8_t spike_flag=0;
    if(!spike_done&&t>=TEST_SPIKE_TIME_S){meas=TEST_SPIKE_VALUE;spike_flag=1;spike_done=true;}

    float fx=FixedKF_Update(&fixedFilter,meas);
    float ax=Kalman_Update(&tempFilter,meas);
    stats_accumulate(sid,truth,meas,fx,ax);

    /* Synthetic timestamp */
    char hms[10];
    format_synthetic_hms(hms,sizeof(hms),t);

    /* Single-buffer CSV row */
    csv_reset();
    csv_field_f(t,3);
    csv_field_s(hms);
    csv_field_u(sid);
    csv_field_f(truth,3);
    csv_field_f(meas,3);
    csv_field_f(fx,3);
    csv_field_f(ax,3);
    csv_field_f(safe_float(tempFilter.Q),5);
    csv_field_f(safe_float(tempFilter.R),5);
    csv_field_f(safe_float(tempFilter.last_K),5);
    csv_field_f(safe_float(tempFilter.P),5);
    csv_field_f(safe_float(tempFilter.last_rate),4);
    csv_append_uint(spike_flag);
    Serial.println(g_csv);
    Serial.flush();

    if(now-last_lcd>=(unsigned long)LCD_REFRESH_INTERVAL_MS){
        last_lcd=now;
#if FEATURE_LCD
        char l0[21],l1[21],l2[21],l3[21];
        snprintf(l0,21,"KF Testbench        ");
        snprintf(l1,21,"t=%4ds Scn=%d       ",(int)t,sid);
        snprintf(l2,21,"Tr=%2d.%d Fx=%2d.%d    ",
                 (int)truth,abs((int)(truth*10)%10),
                 (int)fx,abs((int)(fx*10)%10));
        snprintf(l3,21,"Ad=%2d.%d %s  ",
                 (int)ax,abs((int)(ax*10)%10),hms);
        lcd_safe_update(l0,l1,l2,l3);
#endif
    }
}
#endif

/* ================================================================
 *                    ENERGY TESTBENCH LOOP  (v0.4)
 *
 *  PHASE 1 — Pure computation (no I/O)
 *  PHASE 2 — Build CSV row into buffer (no I/O)
 *  PHASE 3 — Print buffer + flush (UART only, zero I²C)
 *  PHASE 4 — LCD update if due (I²C only, zero UART)
 *
 *  Phases 3 and 4 are NEVER concurrent.  This is the structural
 *  guarantee that eliminates UART/I²C interference.
 * ================================================================ */
#if FEATURE_ENERGY_TESTBENCH
void loop_energy_testbench(void)
{
    static bool          first_tick = true;
    static unsigned long start_time = 0;
    static unsigned long last_sample = 0;
    static unsigned long last_lcd = 0;
    static bool          finished = false;
    static uint32_t      sample_idx = 0;

    if(finished) return;

    unsigned long now = millis();
    if(first_tick){ start_time = now; first_tick = false; }
    if(now - last_sample < (unsigned long)ENERGY_SAMPLE_MS) return;
    last_sample = now;

    /* Deterministic logical time */
    const float dt_s = ENERGY_SAMPLE_MS / 1000.0f;
    float t = (float)sample_idx * dt_s;
    sample_idx++;

    /* End-of-experiment */
    if(t > ENERGY_DURATION_S){
        estats_report();
        Serial.println(F("# === Energy experiment complete ==="));
        Serial.flush();
#if FEATURE_LCD
        delay(5);
        lcd_clear();
        lcd_set_cursor(0,0); lcd_print("TinyEP-Lite DONE");
        lcd_set_cursor(1,0); lcd_print("See serial CSV. ");
#endif
        finished = true;
        return;
    }

    /* ════════ PHASE 1: Pure computation ════════ */

    /* Synthetic timestamp (FIX-A: no I²C) */
    char hms[10];
    format_synthetic_hms(hms, sizeof(hms), t);

    /* Physics */
    float p_truth = power_truth(t);
    float temp_c  = environment_temp(t);

    /* Simulated ADC */
    uint8_t  clip_flag = 0;
    uint16_t code = adc_measure_power(p_truth, temp_c, &clip_flag);
    float    p_raw = adc_to_power_raw(code);
    if(clip_flag) estats.clip_events++;

    /* Filtering */
    float p_kf    = FixedKF_Update(&powerFixedKF, p_raw);
    float p_adapt = adaptive_power_estimate(p_raw);
    float p_combo = Kalman_Update(&powerAdaptKF, p_adapt);

    /* Validity gate */
    bool ok = !isnan(p_raw)   && !isinf(p_raw)   && p_raw   >= 0.0f
           && !isnan(p_kf)    && !isinf(p_kf)    && p_kf    >= 0.0f
           && !isnan(p_adapt) && !isinf(p_adapt) && p_adapt >= 0.0f
           && !isnan(p_combo) && !isinf(p_combo) && p_combo >= 0.0f;

    /* Energy integration */
    E_truth += (double)p_truth * dt_s;
    if(ok){
        E_raw   += (double)p_raw   * dt_s;
        E_kf    += (double)p_kf    * dt_s;
        E_adapt += (double)p_adapt * dt_s;
        E_combo += (double)p_combo * dt_s;
    } else {
        estats.bad_samples++;
    }

    /* Feedback window */
    fb_E_truth_win += (double)p_truth * dt_s;
    if(ok){
        fb_E_est_win   += (double)p_adapt * dt_s;
        fb_sum_raw_win += (double)p_raw   * dt_s;
    }
    fb_T_win += dt_s;

    /* Sparse feedback */
    uint8_t feedback_event = 0;
    if((t - fb_last_time_s) >= FUEL_PERIOD_S){
        sparse_feedback_update(fb_E_truth_win, fb_E_est_win,
                               fb_sum_raw_win, fb_T_win);
        fb_E_truth_win = fb_E_est_win = fb_sum_raw_win = 0;
        fb_T_win = 0;
        fb_last_time_s = t;
        feedback_event = 1;
        estats.fb_events++;
    }

    /* Statistics */
    if(ok) estats_accumulate(p_truth, p_raw, p_kf, p_adapt, p_combo);

    /* Pre-compute safe values */
    float sf_pt = safe_float(p_truth);
    float sf_pr = safe_float(p_raw);
    float sf_pk = safe_float(p_kf);
    float sf_pa = safe_float(p_adapt);
    float sf_pc = safe_float(p_combo);
    float sf_et = safe_d(E_truth);
    float sf_er = safe_d(E_raw);
    float sf_ek = safe_d(E_kf);
    float sf_ea = safe_d(E_adapt);
    float sf_ec = safe_d(E_combo);
    float sf_a  = safe_float(affine_a);
    float sf_b  = safe_float(affine_b);

    /* ════════ PHASE 2: Build CSV row into single buffer ════════ */

    csv_reset();
    csv_field_f(t, 3);            /* t_s       */
    csv_field_s(hms);             /* rtc_hms   */
    csv_field_f(temp_c, 1);       /* temp_c    */
    csv_field_f(sf_pt, 5);        /* truth_W   */
    csv_field_u(code);            /* adc       */
    csv_field_f(sf_pr, 5);        /* raw_W     */
    csv_field_f(sf_pk, 5);        /* kf_W      */
    csv_field_f(sf_pa, 5);        /* adapt_W   */
    csv_field_f(sf_pc, 5);        /* combo_W   */
    csv_field_f(sf_et, 4);        /* E_truth_J */
    csv_field_f(sf_er, 4);        /* E_raw_J   */
    csv_field_f(sf_ek, 4);        /* E_kf_J    */
    csv_field_f(sf_ea, 4);        /* E_adapt_J */
    csv_field_f(sf_ec, 4);        /* E_combo_J */
    csv_field_f(sf_a,  5);        /* affine_a  */
    csv_field_f(sf_b,  5);        /* affine_b  */
    csv_field_u(clip_flag);       /* clip      */
    csv_append_uint(feedback_event); /* fb (last field, no trailing comma) */

    /* ════════ PHASE 3: Atomic serial output ════════ */

    Serial.println(g_csv);
    Serial.flush();    /* FULLY drain TX before anything else happens */

    /* ════════ PHASE 4: LCD update (isolated I²C phase) ════════ */

    if(now - last_lcd >= (unsigned long)LCD_REFRESH_INTERVAL_MS){
        last_lcd = now;
#if FEATURE_LCD
        delay(5);  /* extra safety gap after flush */
        char l0[21], l1[21], l2[21], l3[21];
        snprintf(l0, 21, "TinyEP-Lite  v0.4   ");
        snprintf(l1, 21, "t=%4ds T=%2dC fb=%2u  ",
                 (int)t, (int)temp_c, (unsigned)estats.fb_events);
        int pt_mw = (int)(sf_pt * 1000); if(pt_mw < 0) pt_mw = 0;
        int pc_mw = (int)(sf_pc * 1000); if(pc_mw < 0) pc_mw = 0;
        snprintf(l2, 21, "Pt=%4dmW Pc=%4dmW  ", pt_mw, pc_mw);
        int et_j = (int)sf_et; if(et_j < 0) et_j = 0;
        int ec_j = (int)sf_ec; if(ec_j < 0) ec_j = 0;
        snprintf(l3, 21, "Et=%3dJ Ec=%3dJ %s ", et_j, ec_j, hms);
        /* lcd_safe_update does flush+delay internally */
        lcd_safe_update(l0, l1, l2, l3);
#endif
    }
}
#endif /* FEATURE_ENERGY_TESTBENCH */

/* ================================================================
 *                    LIVE-DEMO LOOP (DHT22)
 * ================================================================ */
#if !FEATURE_TESTBENCH && !FEATURE_ENERGY_TESTBENCH
void loop_live(){
    static bool first_tick=true;
    static unsigned long start_time=0, last_read=0, last_lcd=0;
    static uint32_t ok_count=0, fail_count=0;
    static bool header_done=false;

    if(!header_done){
        Serial.println();
        Serial.println(F("t_s,rtc_hms,raw_C,fixed_C,adapt_C,humidity,status"));
        header_done=true;
    }

    unsigned long now=millis();
    if(first_tick){start_time=now;first_tick=false;}

    if(now-last_read>=2000UL){
        last_read=now;
        float t=(now-start_time)/1000.0f;
        float new_t=0, new_h=0;
        bool rd_ok=dht22_read(&new_t,&new_h);
        if(rd_ok&&(isnan(new_t)||isinf(new_t)||new_t<-40||new_t>80))
            rd_ok=false;

        char hms[10];
        format_synthetic_hms(hms, sizeof(hms), t);

        if(rd_ok){
            ok_count++;
            rawTemperature=new_t; rawHumidity=new_h;
            filteredTemperature=Kalman_Update(&tempFilter,rawTemperature);
            fixedTemperature=FixedKF_Update(&fixedFilter,rawTemperature);

            csv_reset();
            csv_field_f(t,2);
            csv_field_s(hms);
            csv_field_f(safe_float(rawTemperature),2);
            csv_field_f(safe_float(fixedTemperature),2);
            csv_field_f(safe_float(filteredTemperature),2);
            csv_field_f(rawHumidity,1);
            csv_append_str("OK");
            Serial.println(g_csv);
            Serial.flush();
        } else {
            fail_count++;
            csv_reset();
            csv_field_f(t,2);
            csv_field_s(hms);
            csv_append_str(",,,,,FAIL");
            Serial.println(g_csv);
            Serial.flush();
        }
    }

    if(now-last_lcd>=(unsigned long)LCD_REFRESH_INTERVAL_MS){
        last_lcd=now;
#if FEATURE_LCD
        char l0[21],l1[21],l2[21],l3[21];
        snprintf(l0,21,"=Kalman Live Demo=  ");
        snprintf(l1,21,"Raw:  %d.%d C        ",
                 (int)rawTemperature,abs((int)(rawTemperature*10)%10));
        snprintf(l2,21,"Filt: %d.%02d C       ",
                 (int)filteredTemperature,abs((int)(filteredTemperature*100)%100));
        snprintf(l3,21,"OK:%-4lu FAIL:%-4lu   ",
                 (unsigned long)ok_count,(unsigned long)fail_count);
        lcd_safe_update(l0,l1,l2,l3);
#endif
    }
}
#endif

/* ================================================================
 *                    MAIN LOOP DISPATCH
 * ================================================================ */
void loop(){
#if   FEATURE_ENERGY_TESTBENCH
    loop_energy_testbench();
#elif FEATURE_TESTBENCH
    loop_testbench();
#else
    loop_live();
#endif
}