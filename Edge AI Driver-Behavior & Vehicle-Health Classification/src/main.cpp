/****************************************************************************
 * AI-Powered Driving Behavior Classifier  –  STM32F103 Blue Pill
 * Author : Mohamed Abdelnasser Mehery
 *
 *  MODE 0 = TRAIN : streams CSV on Serial for offline ML training
 *  MODE 1 = INFER : runs TinyAI MLP, drives OLED / LEDs / buzzer / relay
 ****************************************************************************/
#define APP_MODE 0        // 0 = TRAIN (CSV)  |  1 = INFER (TinyAI)

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

/* ── Pin map ───────────────────────────────────────────────────────────── */
#define PIN_NTC    PA0
#define PIN_POT    PA1
#define PIN_RELAY  PB12
#define PIN_BUZZER PB13
#define PIN_BTN    PB14
#define LED_NORMAL PB0
#define LED_ECO    PB1
#define LED_FAULT  PA8

/* ── Config ────────────────────────────────────────────────────────────── */
#define N_IN       8
#define N_OUT      3
#define WIN_SIZE   20
#define SAMPLE_MS  50

Adafruit_SSD1306 oled(128, 64, &Wire, -1);

/* ── MPU6050 raw I2C (NO repeated start, with timeout) ────────────────── */
#define MPU_ADDR        0x68
#define MPU_REG_PWR1    0x6B
#define MPU_REG_ACFG    0x1C
#define MPU_REG_GCFG    0x1B
#define MPU_REG_ACCEL   0x3B
#define MPU_REG_WHOAMI  0x75

bool mpuReady = false;

static void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(true);          // full STOP
}

/* Wait for I2C bytes with a hard timeout so we never hang */
static bool waitBytes(uint8_t n, uint32_t ms) {
  uint32_t t0 = millis();
  while (Wire.available() < n) {
    if (millis() - t0 > ms) return false;
  }
  return true;
}

static uint8_t mpuReadByte(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(true) != 0) return 0xFF;   // STOP, not repeated start
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
  if (!waitBytes(1, 50)) return 0xFF;
  return Wire.read();
}

static bool mpuReadBytes(uint8_t reg, uint8_t *buf, uint8_t n) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(true) != 0) return false;
  Wire.requestFrom((uint8_t)MPU_ADDR, n);
  if (!waitBytes(n, 50)) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

static bool mpuInit() {
  uint8_t who = mpuReadByte(MPU_REG_WHOAMI);
  Serial.print("  WHO_AM_I = 0x"); Serial.println(who, HEX); Serial.flush();
  if (who == 0xFF) return false;       // bus dead
  // Wokwi MPU returns 0x68 or 0x72 depending on revision; accept anything plausible
  mpuWrite(MPU_REG_PWR1, 0x00);  delay(20);
  mpuWrite(MPU_REG_ACFG, 0x10);  delay(5);
  mpuWrite(MPU_REG_GCFG, 0x08);  delay(5);
  return true;
}

static void mpuRead(float &ax, float &ay, float &az, float &gz) {
  uint8_t b[14];
  if (!mpuReadBytes(MPU_REG_ACCEL, b, 14)) {
    ax = ay = 0; az = 9.81f; gz = 0; return;
  }
  int16_t rax = (int16_t)((b[0]  << 8) | b[1]);
  int16_t ray = (int16_t)((b[2]  << 8) | b[3]);
  int16_t raz = (int16_t)((b[4]  << 8) | b[5]);
  int16_t rgz = (int16_t)((b[12] << 8) | b[13]);
  ax = rax / 4096.0f * 9.81f;
  ay = ray / 4096.0f * 9.81f;
  az = raz / 4096.0f * 9.81f;
  gz = rgz / 65.5f;
}

/* ── Buffers ───────────────────────────────────────────────────────────── */
float bufLong[WIN_SIZE], bufLat[WIN_SIZE], bufMag[WIN_SIZE], bufYaw[WIN_SIZE];
int   widx = 0;
const char* CLASS_NAME[N_OUT] = {"ECO", "NORMAL", "AGGRESSIVE"};

/* =========================================================================
 *  ████  TINYAI PARAMS START  ████   ← REPLACE after training
 * ========================================================================= */

#define N_H 8

const float norm_mean[8] = {-3.597752,0.473106,4.449901,0.262090,0.087425,4.856888,11.274636,14.717204};
const float norm_scale[8] = {7.025675,1.463890,2.907080,0.832591,0.221173,13.154801,4.934655,10.565219}; 

const float W1[8][8] = {
  {0.240790,0.471248,1.192340,-0.872688,-0.262797,-0.101430,-1.548336,0.714860},
  {1.124481,0.658304,-0.106039,0.120582,0.411808,0.818566,0.636686,0.405305},
  {0.028318,0.158301,0.229437,-0.471818,0.367469,-0.001889,0.054292,1.700398},
  {-1.924127,-0.179678,0.541425,0.283683,-0.629387,-1.403331,-1.134564,-0.191002},
  {1.120642,0.310885,1.116545,-0.216266,-0.096017,-1.271692,0.633699,0.182994},
  {-1.987148,0.473763,1.058954,-0.210669,-0.364567,-0.292055,0.314171,-0.709513},
  {2.658438,0.989380,0.042595,-0.807581,-0.104036,0.009281,-0.000362,-0.668534},
  {0.533486,-1.066488,-0.732784,0.465055,-0.349295,0.883685,-0.400013,-1.286648}
};
const float b1[8] = {0.350819,-0.581885,0.203542,1.082043,-0.704105,0.325742,-1.329304,0.333967}; 

const float W2[3][8] = {
  {0.128389,-1.020677,0.886590,0.356197,-1.873585,0.822461,1.948494,1.074712},
  {-1.665476,1.496869,0.323129,1.832824,0.559689,-2.083906,0.715412,-1.787952},
  {1.586315,-0.332454,-1.552288,-2.050949,0.980832,1.097146,-2.521588,0.489919}
};
const float b2[3] = {-2.482683,1.509986,0.972697};
/* =========================================================================
 *  ████  TINYAI PARAMS END  ████
 * ========================================================================= */

static inline float reluf(float x) { return x > 0.f ? x : 0.f; }

float readTempC() {
  int raw = analogRead(PIN_NTC);
  if (raw < 5)    return -40.0f;
  if (raw > 4090) return 125.0f;
  float v = raw / 4095.0f;
  float r = 10000.0f * v / (1.0f - v);
  float tK = 1.0f / (1.0f/298.15f + logf(r/10000.0f)/3950.0f);
  return tK - 273.15f;
}

int tinyAI(const float feat[N_IN], float probs[N_OUT]) {
  float xn[N_IN], h[N_H], o[N_OUT];
  for (int i = 0; i < N_IN; i++) {
    float sc = norm_scale[i]; if (fabsf(sc) < 1e-6f) sc = 1.0f;
    xn[i] = (feat[i] - norm_mean[i]) / sc;
  }
  for (int j = 0; j < N_H; j++) {
    float s = b1[j];
    for (int i = 0; i < N_IN; i++) s += W1[j][i] * xn[i];
    h[j] = reluf(s);
  }
  float mx = -1e30f;
  for (int k = 0; k < N_OUT; k++) {
    float s = b2[k];
    for (int j = 0; j < N_H; j++) s += W2[k][j] * h[j];
    o[k] = s; if (s > mx) mx = s;
  }
  float sum = 0;
  for (int k = 0; k < N_OUT; k++) { o[k] = expf(o[k]-mx); sum += o[k]; }
  int best = 0;
  for (int k = 0; k < N_OUT; k++) {
    probs[k] = o[k] / sum;
    if (probs[k] > probs[best]) best = k;
  }
  return best;
}

void computeFeatures(float f[N_IN]) {
  float mL=0,mLat=0,mMag=0,mYaw=0,sL=0,sLat=0,sYaw=0,maxMag=0,jerk=0;
  for (int i = 0; i < WIN_SIZE; i++) {
    mL+=bufLong[i]; mLat+=fabsf(bufLat[i]); mMag+=bufMag[i]; mYaw+=bufYaw[i];
  }
  mL/=WIN_SIZE; mLat/=WIN_SIZE; mMag/=WIN_SIZE; mYaw/=WIN_SIZE;
  for (int i = 0; i < WIN_SIZE; i++) {
    sL   += (bufLong[i]-mL)*(bufLong[i]-mL);
    sLat += (fabsf(bufLat[i])-mLat)*(fabsf(bufLat[i])-mLat);
    sYaw += (bufYaw[i]-mYaw)*(bufYaw[i]-mYaw);
    if (bufMag[i] > maxMag) maxMag = bufMag[i];
    if (i) jerk += fabsf(bufMag[i]-bufMag[i-1]);
  }
  f[0] = mL;
  f[1] = sqrtf(fmaxf(sL  / WIN_SIZE, 0.0f));
  f[2] = mLat;
  f[3] = sqrtf(fmaxf(sLat/ WIN_SIZE, 0.0f));
  f[4] = jerk/WIN_SIZE;
  f[5] = sqrtf(fmaxf(sYaw/ WIN_SIZE, 0.0f));
  f[6] = maxMag;
  f[7] = readTempC();
}

void showOLED(int cls, float conf, float thr) {
  oled.clearDisplay();
  oled.setTextSize(1); oled.setCursor(0,0);
  oled.println("DRIVE-AI (edge)");
  oled.drawFastHLine(0,10,128,SSD1306_WHITE);
  oled.setTextSize(2); oled.setCursor(0,18);
  oled.println(CLASS_NAME[cls]);
  oled.setTextSize(1); oled.setCursor(0,42);
  oled.print("conf: "); oled.print(conf*100,0); oled.println("%");
  oled.print("thr : "); oled.print(thr*100,0);  oled.println("%");
  oled.display();
}

/* ── setup ─────────────────────────────────────────────────────────────── */
void setup() {
  Serial.begin(115200);
  delay(400);
  Serial.println();
  Serial.println("=== DRIVE-AI BOOT ==="); Serial.flush();

  Serial.println("[1] pins"); Serial.flush();
  pinMode(PIN_RELAY, OUTPUT);  digitalWrite(PIN_RELAY, LOW);
  pinMode(PIN_BUZZER, OUTPUT); digitalWrite(PIN_BUZZER, LOW);
  pinMode(LED_NORMAL, OUTPUT); digitalWrite(LED_NORMAL, LOW);
  pinMode(LED_ECO, OUTPUT);    digitalWrite(LED_ECO, LOW);
  pinMode(LED_FAULT, OUTPUT);  digitalWrite(LED_FAULT, LOW);
  pinMode(PIN_BTN, INPUT_PULLUP);
  analogReadResolution(12);

  Serial.println("[2] Wire.begin"); Serial.flush();
  Wire.begin();
  Wire.setClock(100000);          // safe 100 kHz
  delay(100);

  /* OLED FIRST – we know it works */
  Serial.println("[3] OLED.begin"); Serial.flush();
  if (!oled.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("    OLED FAIL");
  } else {
    Serial.println("    OLED OK");
    oled.clearDisplay();
    oled.setTextColor(SSD1306_WHITE);
    oled.setTextSize(1); oled.setCursor(0,0);
    oled.println("DRIVE-AI READY");
    oled.display();
  }
  Serial.flush();

  /* MPU SECOND – with timeouts so it can't hang */
  Serial.println("[4] MPU init"); Serial.flush();
  mpuReady = mpuInit();
  Serial.println(mpuReady ? "    MPU6050 OK" : "    MPU6050 FAIL (using fake data)");
  Serial.flush();

  Serial.print("[5] APP_MODE = ");
  Serial.println(APP_MODE == 0 ? "TRAIN (CSV)" : "INFER (TinyAI)");
  Serial.flush();

#if APP_MODE == 0
  Serial.println("f0,f1,f2,f3,f4,f5,f6,f7,label");
#endif
  Serial.println("SETUP DONE — data follows");
  Serial.println("--------------------------------");
  Serial.flush();
}

/* ── loop ──────────────────────────────────────────────────────────────── */
void loop() {
  float ax = 0, ay = 0, az = 9.81f, gz = 0;
  if (mpuReady) {
    mpuRead(ax, ay, az, gz);
  } else {
    // Generate fake but varying data so CSV is still useful for testing
    float t = millis() / 1000.0f;
    ax = 0.5f * sinf(t * 1.7f);
    ay = 0.3f * sinf(t * 2.3f);
    gz = 5.0f * sinf(t * 0.9f);
  }

  bufLong[widx] = ax;
  bufLat[widx]  = ay;
  bufMag[widx]  = sqrtf(ax*ax + ay*ay + az*az);
  bufYaw[widx]  = gz;
  widx++;
  delay(SAMPLE_MS);

  if (widx < WIN_SIZE) return;
  widx = 0;

  float feat[N_IN];
  computeFeatures(feat);

#if APP_MODE == 0
  int pot   = analogRead(PIN_POT);
  int label = (pot < 1365) ? 0 : (pot < 2730) ? 1 : 2;

  // Sanitize: replace NaN/Inf with 0.0
  for (int i = 0; i < N_IN; i++) {
    if (isnan(feat[i]) || isinf(feat[i])) feat[i] = 0.0f;
    // Clamp extreme values
    if (feat[i] >  1000.0f) feat[i] =  1000.0f;
    if (feat[i] < -1000.0f) feat[i] = -1000.0f;
  }

  // Print using Arduino's built-in Serial.print 
  // (snprintf float support is disabled on STM32 by default)
  for (int i = 0; i < N_IN; i++) {
    Serial.print(feat[i], 4);
    Serial.print(",");
  }
  Serial.println(label);
  Serial.flush(); // Ensure the whole line is sent before moving on


  digitalWrite(LED_ECO,    label == 0);
  digitalWrite(LED_NORMAL, label == 1);
  digitalWrite(LED_FAULT,  label == 2);

  oled.clearDisplay();
  oled.setTextSize(1); oled.setCursor(0,0);
  oled.println("REC -> CSV");
  oled.setTextSize(2); oled.setCursor(0,18);
  oled.println(CLASS_NAME[label]);
  oled.setTextSize(1); oled.setCursor(0,48);
  oled.print("T=");    oled.print(feat[7],1);
  oled.print(" pk="); oled.print(feat[6],1);
  oled.display();

#else
  float probs[N_OUT];
  int   cls  = tinyAI(feat, probs);
  float conf = probs[cls];
  float thr  = analogRead(PIN_POT) / 4095.0f;

  digitalWrite(LED_ECO,    cls == 0);
  digitalWrite(LED_NORMAL, cls == 1);
  digitalWrite(LED_FAULT,  cls == 2);

  bool danger = (cls == 2 && conf > 0.7f && thr > 0.6f);
  digitalWrite(PIN_RELAY,  danger);
  digitalWrite(PIN_BUZZER, danger);

  showOLED(cls, conf, thr);

  Serial.print("CLASS="); Serial.print(CLASS_NAME[cls]);
  Serial.print("  conf="); Serial.print(conf,2);
  if (danger) Serial.print("  [!! LIMP MODE !!]");
  Serial.println();
  Serial.flush();
#endif
}