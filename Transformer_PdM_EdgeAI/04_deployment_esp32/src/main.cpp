/****************************************************************************
 * Transformer Predictive-Maintenance Edge Node — ESP32
 * ====================================================
 *
 *  Reads the transformer's electrical + thermal sensors, rebuilds the same
 *  23-feature window the model was trained on, runs an int8 TFLite-Micro CNN
 *  and produces:
 *
 *      1. a health state       NORMAL / WARNING / CRITICAL (2 h ahead)
 *      2. an ALARM bit         risk = P(WARN)+P(CRIT) >= threshold
 *      3. a WTI forecast       winding temperature in 2 hours, in degC
 *
 *  The alarm threshold, normalisation constants and the model itself all come
 *  from the training pipeline via transformer_pdm_model.h — nothing is
 *  hand-tuned on the device.
 *
 *  MODE 0 = REPLAY : streams the bundled CSV rows through the model.  Use this
 *                    in Wokwi / on the bench to prove the firmware reproduces
 *                    the host results exactly.
 *  MODE 1 = LIVE   : reads real sensors (ADC / one-wire).  Wire up read_sensors().
 *
 *  Author : Mohamed Abdelnasser Mehery
 ****************************************************************************/
#define APP_MODE 0       /* 0 = REPLAY bundled samples | 1 = LIVE sensors */

#include <Arduino.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "transformer_pdm_model.h"
#include "pdm_features.h"
#if APP_MODE == 0
#include "replay_data.h"
#endif

/* ── Pin map (LIVE mode) ─────────────────────────────────────────────── */
#define PIN_CT_L1     34      /* SCT-013 current transformer, phase 1 */
#define PIN_CT_L2     35
#define PIN_CT_L3     32
#define PIN_PT_L1     33      /* ZMPT101B voltage transformer, phase 1 */
#define PIN_PT_L2     25
#define PIN_PT_L3     26
#define PIN_OIL_TEMP  27      /* PT100 / DS18B20 — top oil     */
#define PIN_WIND_TEMP 14      /* PT100          — winding      */
#define PIN_AMB_TEMP  12      /* DS18B20        — ambient      */
#define PIN_OIL_LEVEL 13      /* level sender                  */
#define PIN_LED_OK     2
#define PIN_LED_WARN   4
#define PIN_RELAY     16      /* trip / annunciator            */
#define PIN_BUZZER    17

/* ── Node configuration ──────────────────────────────────────────────── */
static const float I_RATED = 1449.0f;   /* rated phase current, A  */
#define SAMPLE_PERIOD_MS  (15UL * 60UL * 1000UL)   /* 15 min, matches training */

/* In REPLAY mode we do not want to wait 15 real minutes per sample. */
#if APP_MODE == 0
#define TICK_MS   250UL
#else
#define TICK_MS   SAMPLE_PERIOD_MS
#endif

/* ── TFLite-Micro plumbing ───────────────────────────────────────────── */
namespace {
tflite::MicroErrorReporter  g_error_reporter;
const tflite::Model        *g_model = nullptr;
tflite::MicroInterpreter   *g_interp = nullptr;
TfLiteTensor               *g_input = nullptr;
TfLiteTensor               *g_out_cls = nullptr;
TfLiteTensor               *g_out_reg = nullptr;

/* The arena must outlive the interpreter; 8-byte aligned. */
alignas(16) uint8_t g_arena[PDM_TENSOR_ARENA];

pdm_ctx_t g_ctx;
uint32_t  g_last_tick = 0;
uint32_t  g_infer_count = 0;
uint64_t  g_infer_us_total = 0;
}  // namespace

/* ── Helpers ─────────────────────────────────────────────────────────── */
static void fatal(const char *msg)
{
    Serial.printf("{\"fatal\":\"%s\"}\n", msg);
    while (true) {
        digitalWrite(PIN_LED_WARN, !digitalRead(PIN_LED_WARN));
        delay(200);
    }
}

/*
 * Quantise one normalised float into the model's int8 input domain.
 *   q = round(x / scale) + zero_point,  clamped to [-128, 127]
 */
static inline int8_t quantise(float x, float scale, int zp)
{
    int32_t q = (int32_t)lroundf(x / scale) + zp;
    if (q < -128) q = -128;
    if (q >  127) q =  127;
    return (int8_t)q;
}

static inline float dequantise(int8_t q, float scale, int zp)
{
    return ((int32_t)q - zp) * scale;
}

/* ── Sensor acquisition ──────────────────────────────────────────────── */
#if APP_MODE == 1
/*
 * LIVE mode.  Replace the bodies with your real drivers.
 *
 * IMPORTANT: the model was trained on 15-minute AVERAGES, not instantaneous
 * readings.  Sample the AC channels fast (a few kHz for at least 10 cycles),
 * compute true RMS, and average the result over the whole 15-minute period.
 * Feeding a single instantaneous sample here is the fastest way to make a
 * perfectly good model look broken.
 */
static float read_rms_current(int pin)
{
    const int N = 1000;
    double acc = 0.0;
    for (int i = 0; i < N; i++) {
        int raw = analogRead(pin) - 2048;          /* centred on mid-rail */
        double amps = raw * (100.0 / 2048.0);      /* 100 A full scale    */
        acc += amps * amps;
        delayMicroseconds(200);                    /* ~5 kHz              */
    }
    return (float)sqrt(acc / N);
}

static float read_rms_voltage(int pin)
{
    const int N = 1000;
    double acc = 0.0;
    for (int i = 0; i < N; i++) {
        int raw = analogRead(pin) - 2048;
        double volts = raw * (400.0 / 2048.0);     /* 400 V full scale    */
        acc += volts * volts;
        delayMicroseconds(200);
    }
    return (float)sqrt(acc / N);
}

static float read_temperature(int pin)
{
    /* Placeholder: swap for DallasTemperature / MAX31865 PT100 driver. */
    int raw = analogRead(pin);
    return raw * (150.0f / 4095.0f);               /* 0..150 degC span    */
}

static void read_sensors(pdm_sample_t *s)
{
    s->vl1 = read_rms_voltage(PIN_PT_L1);
    s->vl2 = read_rms_voltage(PIN_PT_L2);
    s->vl3 = read_rms_voltage(PIN_PT_L3);
    s->il1 = read_rms_current(PIN_CT_L1);
    s->il2 = read_rms_current(PIN_CT_L2);
    s->il3 = read_rms_current(PIN_CT_L3);
    s->oti = read_temperature(PIN_OIL_TEMP);
    s->wti = read_temperature(PIN_WIND_TEMP);
    s->ati = read_temperature(PIN_AMB_TEMP);
    s->oli = analogRead(PIN_OIL_LEVEL) * (100.0f / 4095.0f);
}
#else
static size_t g_replay_idx = 0;
static bool read_sensors(pdm_sample_t *s)
{
    if (g_replay_idx >= REPLAY_N) return false;
    const float *r = replay_rows[g_replay_idx++];
    s->vl1 = r[0]; s->vl2 = r[1]; s->vl3 = r[2];
    s->il1 = r[3]; s->il2 = r[4]; s->il3 = r[5];
    s->oti = r[6]; s->wti = r[7]; s->ati = r[8]; s->oli = r[9];
    return true;
}
#endif

/* ── Inference ───────────────────────────────────────────────────────── */
struct pdm_result_t {
    float    prob[PDM_N_CLASSES];
    float    risk;
    uint8_t  cls;
    bool     alarm;
    float    wti_forecast;
    uint32_t micros;
};

static bool run_inference(pdm_result_t *out)
{
    const float in_scale = g_input->params.scale;
    const int   in_zp    = g_input->params.zero_point;

    /* normalise -> quantise -> flatten, exactly as in training */
    int8_t *dst = g_input->data.int8;
    for (int t = 0; t < PDM_WINDOW; t++) {
        for (int f = 0; f < PDM_N_FEATURES; f++) {
            float x = (g_ctx.feat[t][f] - pdm_norm_mean[f]) / pdm_norm_scale[f];
            *dst++ = quantise(x, in_scale, in_zp);
        }
    }

    const uint32_t t0 = micros();
    if (g_interp->Invoke() != kTfLiteOk) return false;
    out->micros = micros() - t0;

    const float cs = g_out_cls->params.scale;
    const int   cz = g_out_cls->params.zero_point;
    float sum = 0.0f;
    for (int i = 0; i < PDM_N_CLASSES; i++) {
        out->prob[i] = dequantise(g_out_cls->data.int8[i], cs, cz);
        sum += out->prob[i];
    }
    if (sum > 1e-6f) for (int i = 0; i < PDM_N_CLASSES; i++) out->prob[i] /= sum;

    out->cls = 0;
    for (int i = 1; i < PDM_N_CLASSES; i++)
        if (out->prob[i] > out->prob[out->cls]) out->cls = (uint8_t)i;

    out->risk  = out->prob[1] + out->prob[2];
    out->alarm = (out->risk >= PDM_ALARM_THRESH);

    const float rs = g_out_reg->params.scale;
    const int   rz = g_out_reg->params.zero_point;
    out->wti_forecast =
        dequantise(g_out_reg->data.int8[0], rs, rz) * PDM_WTI_STD + PDM_WTI_MEAN;

    g_infer_count++;
    g_infer_us_total += out->micros;
    return true;
}

static void drive_outputs(const pdm_result_t &r)
{
    digitalWrite(PIN_LED_OK,   r.alarm ? LOW : HIGH);
    digitalWrite(PIN_LED_WARN, r.alarm ? HIGH : LOW);
    /* Only a predicted CRITICAL state actuates the relay; a WARNING just
     * annunciates.  Tripping a live transformer on a soft warning would be
     * worse than the fault. */
    digitalWrite(PIN_RELAY, (r.alarm && r.cls == 2) ? HIGH : LOW);
    if (r.alarm && r.cls == 2) tone(PIN_BUZZER, 2000, 250);
}

/* ── Arduino entry points ────────────────────────────────────────────── */
void setup()
{
    Serial.begin(115200);
    delay(300);

    pinMode(PIN_LED_OK, OUTPUT);
    pinMode(PIN_LED_WARN, OUTPUT);
    pinMode(PIN_RELAY, OUTPUT);
    pinMode(PIN_BUZZER, OUTPUT);
    digitalWrite(PIN_RELAY, LOW);

#if APP_MODE == 1
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
#endif

    Serial.println();
    Serial.println(F("=== Transformer PdM edge node (ESP32) ==="));
    Serial.printf("model %u bytes | window %d x %d | horizon %d samples\n",
                  transformer_pdm_model_len, PDM_WINDOW, PDM_N_FEATURES, PDM_HORIZON);

    g_model = tflite::GetModel(transformer_pdm_model);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) fatal("schema mismatch");

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter interp(g_model, resolver, g_arena,
                                           PDM_TENSOR_ARENA, &g_error_reporter);
    g_interp = &interp;

    if (g_interp->AllocateTensors() != kTfLiteOk) fatal("AllocateTensors failed");

    g_input = g_interp->input(0);
    /* output order is not guaranteed — identify the heads by their size */
    TfLiteTensor *o0 = g_interp->output(0);
    TfLiteTensor *o1 = g_interp->output(1);
    const int o0n = o0->dims->data[o0->dims->size - 1];
    g_out_cls = (o0n == PDM_N_CLASSES) ? o0 : o1;
    g_out_reg = (o0n == PDM_N_CLASSES) ? o1 : o0;

    Serial.printf("arena used   : %u / %u bytes\n",
                  (unsigned)g_interp->arena_used_bytes(), (unsigned)PDM_TENSOR_ARENA);
    Serial.printf("input        : %d dims, scale %.6f zp %d\n",
                  g_input->dims->size, g_input->params.scale, g_input->params.zero_point);
    Serial.printf("alarm thresh : %.3f\n", PDM_ALARM_THRESH);
    Serial.println(F("-----------------------------------------"));

    pdm_init(&g_ctx, I_RATED);
    g_last_tick = millis() - TICK_MS;
}

void loop()
{
    if ((uint32_t)(millis() - g_last_tick) < TICK_MS) return;
    g_last_tick = millis();

    pdm_sample_t s;
#if APP_MODE == 0
    if (!read_sensors(&s)) {
        Serial.printf("{\"done\":true,\"inferences\":%u,\"avg_us\":%.1f}\n",
                      g_infer_count,
                      g_infer_count ? (double)g_infer_us_total / g_infer_count : 0.0);
        delay(10000);
        return;
    }
#else
    read_sensors(&s);
#endif

    pdm_push(&g_ctx, &s);
    /* Gate on the FULL warm-up, not just a full window: the 1-hour trend
     * features are still zero for the first PDMF_LAG_1H samples and the model
     * was never trained on that. */
    if (!pdm_ready(&g_ctx)) {
        Serial.printf("{\"warmup\":%u,\"need\":%d}\n",
                      g_ctx.count, PDM_WARMUP_SAMPLES);
        return;
    }

    pdm_result_t r;
    if (!run_inference(&r)) {
        Serial.println(F("{\"error\":\"invoke failed\"}"));
        return;
    }
    drive_outputs(r);

    /* One JSON line per inference — easy to pipe into a logger or MQTT. */
    Serial.printf(
        "{\"t\":%lu,\"wti\":%.2f,\"oti\":%.2f,\"load\":%.3f,\"sev_now\":%u,"
        "\"pred\":\"%s\",\"risk\":%.3f,\"alarm\":%s,\"wti_fc\":%.2f,\"us\":%lu}\n",
        (unsigned long)millis(), s.wti, s.oti,
        g_ctx.feat[PDM_WINDOW - 1][14], g_ctx.sev_now,
        pdm_class_name[r.cls], r.risk, r.alarm ? "true" : "false",
        r.wti_forecast, (unsigned long)r.micros);
}
