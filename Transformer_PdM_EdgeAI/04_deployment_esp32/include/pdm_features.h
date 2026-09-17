/****************************************************************************
 * pdm_features.h — on-device feature engineering for the transformer PdM model
 *
 * This MUST stay bit-for-bit equivalent to 01_data/preprocess.py::add_features
 * and ::label_health.  If the two ever diverge, the model silently sees a
 * different input distribution than it was trained on and the accuracy quietly
 * collapses — the single most common way an edge ML deployment fails.
 *
 * The parity is enforced by 04_deployment_esp32/scripts/test_parity.py, which
 * compiles this file on the host and diffs its output against the Python
 * pipeline over the whole test split.
 *
 * Author : Mohamed Abdelnasser Mehery
 ****************************************************************************/
#ifndef PDM_FEATURES_H
#define PDM_FEATURES_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- must match config.py ------------------------------------------- */
#define PDMF_WINDOW        16
#define PDMF_N_FEATURES    23
#define PDMF_SAMPLE_MIN    15          /* minutes between samples          */
#define PDMF_LAG_1H        4           /* 60 / PDMF_SAMPLE_MIN             */

/*
 * Warm-up
 * -------
 * WTI_trend_1h / OTI_trend_1h need PDMF_LAG_1H samples of history, and the
 * window itself needs PDMF_WINDOW.  Until BOTH are satisfied the feature
 * vector does not match what the model saw in training (the trend terms read
 * 0), so the first inferences would be quietly wrong.  pdm_ready() gates on
 * this; the firmware must not raise alarms before it returns 1.
 */
#define PDM_WARMUP_SAMPLES (PDMF_WINDOW + PDMF_LAG_1H)

/* thresholds (config.py THRESH) */
#define PDMF_WTI_WARN      95.0f
#define PDMF_WTI_CRIT     110.0f
#define PDMF_OTI_WARN      75.0f
#define PDMF_OTI_CRIT      90.0f
#define PDMF_OLI_WARN      30.0f
#define PDMF_OLI_CRIT      20.0f
#define PDMF_VUNB_WARN      2.0f
#define PDMF_VUNB_CRIT      5.0f
#define PDMF_IUNB_WARN     10.0f
#define PDMF_IUNB_CRIT     25.0f
#define PDMF_LOAD_WARN      1.10f
#define PDMF_LOAD_CRIT      1.30f

/* One raw sensor sample (what the ADC front-end produces). */
typedef struct {
    float vl1, vl2, vl3;   /* phase-neutral volts        */
    float il1, il2, il3;   /* phase amps                 */
    float oti;             /* top-oil temperature   degC */
    float wti;             /* winding temperature   degC */
    float ati;             /* ambient temperature   degC */
    float oli;             /* oil level             %    */
} pdm_sample_t;

/* Rolling history: raw samples + the derived feature matrix fed to the NN. */
typedef struct {
    pdm_sample_t raw[PDMF_WINDOW];
    float feat[PDMF_WINDOW][PDMF_N_FEATURES];
    /* extra tail of OTI/WTI so the 1-hour trend is defined for row 0 */
    float oti_hist[PDMF_LAG_1H + 1];
    float wti_hist[PDMF_LAG_1H + 1];
    float i_rated;         /* rated phase current, A */
    uint16_t count;        /* total samples pushed   */
    uint8_t  sev_now;      /* rule severity of the newest sample */
} pdm_ctx_t;

static inline float pdm_maxf(float a, float b) { return a > b ? a : b; }

static inline void pdm_init(pdm_ctx_t *c, float i_rated)
{
    memset(c, 0, sizeof(*c));
    c->i_rated = (i_rated > 1e-6f) ? i_rated : 1.0f;
}

/* 1 once the window AND the 1-hour trend buffers are fully primed. */
static inline int pdm_ready(const pdm_ctx_t *c)
{
    return c->count >= PDM_WARMUP_SAMPLES;
}

/* NEMA MG-1 unbalance: max deviation from the mean, as % of the mean. */
static inline float pdm_unbalance(float a, float b, float cc)
{
    const float avg = (a + b + cc) / 3.0f;
    if (avg <= 1e-6f) return 0.0f;
    float d1 = fabsf(a - avg), d2 = fabsf(b - avg), d3 = fabsf(cc - avg);
    float mx = pdm_maxf(d1, pdm_maxf(d2, d3));
    return mx / avg * 100.0f;
}

/* Rule-based severity of a single sample — mirrors label_health(). */
static inline uint8_t pdm_severity(float wti, float oti, float oli,
                                   float v_unb, float i_unb, float load_pu)
{
    uint8_t s = 0;
    #define PDMF_BUMP(warn, crit) do {                       \
        uint8_t t = (crit) ? 2u : ((warn) ? 1u : 0u);        \
        if (t > s) s = t;                                    \
    } while (0)
    PDMF_BUMP(wti     >= PDMF_WTI_WARN,  wti     >= PDMF_WTI_CRIT);
    PDMF_BUMP(oti     >= PDMF_OTI_WARN,  oti     >= PDMF_OTI_CRIT);
    PDMF_BUMP(oli     <= PDMF_OLI_WARN,  oli     <= PDMF_OLI_CRIT);
    PDMF_BUMP(v_unb   >= PDMF_VUNB_WARN, v_unb   >= PDMF_VUNB_CRIT);
    PDMF_BUMP(i_unb   >= PDMF_IUNB_WARN, i_unb   >= PDMF_IUNB_CRIT);
    PDMF_BUMP(load_pu >= PDMF_LOAD_WARN, load_pu >= PDMF_LOAD_CRIT);
    #undef PDMF_BUMP
    return s;
}

/*
 * Push one sample, shifting the window.  Returns 1 once the window is full
 * (i.e. an inference may be run), 0 while still filling.
 *
 * Feature order (index -> name) — identical to meta.json["features"]:
 *   0 VL1  1 VL2  2 VL3  3 IL1  4 IL2  5 IL3  6 OTI  7 WTI  8 ATI  9 OLI
 *  10 V_avg 11 I_avg 12 V_unbal 13 I_unbal 14 load_pu 15 dT_oil_amb
 *  16 dT_wind_oil 17 OTI_roc 18 WTI_roc 19 WTI_trend_1h 20 OTI_trend_1h
 *  21 sev_now 22 margin_wti
 */
static inline int pdm_push(pdm_ctx_t *c, const pdm_sample_t *s)
{
    /* previous sample (for the rate-of-change terms) */
    const int had_prev = (c->count > 0);
    const pdm_sample_t prev = c->raw[PDMF_WINDOW - 1];

    /* shift raw + feature ring by one */
    memmove(&c->raw[0], &c->raw[1], sizeof(pdm_sample_t) * (PDMF_WINDOW - 1));
    memmove(&c->feat[0], &c->feat[1], sizeof(float) * PDMF_N_FEATURES * (PDMF_WINDOW - 1));
    c->raw[PDMF_WINDOW - 1] = *s;

    /* shift the 1-hour tail buffers */
    memmove(&c->oti_hist[0], &c->oti_hist[1], sizeof(float) * PDMF_LAG_1H);
    memmove(&c->wti_hist[0], &c->wti_hist[1], sizeof(float) * PDMF_LAG_1H);
    c->oti_hist[PDMF_LAG_1H] = s->oti;
    c->wti_hist[PDMF_LAG_1H] = s->wti;

    const float v_avg = (s->vl1 + s->vl2 + s->vl3) / 3.0f;
    const float i_avg = (s->il1 + s->il2 + s->il3) / 3.0f;
    const float v_unb = pdm_unbalance(s->vl1, s->vl2, s->vl3);
    const float i_unb = pdm_unbalance(s->il1, s->il2, s->il3);
    const float load  = i_avg / c->i_rated;

    /* pandas .diff() yields NaN on the first row, which preprocess.py
     * fills with 0.0 — reproduce that exactly. */
    const float oti_roc = had_prev ? (s->oti - prev.oti) : 0.0f;
    const float wti_roc = had_prev ? (s->wti - prev.wti) : 0.0f;
    const int   have_1h = (c->count >= PDMF_LAG_1H);
    const float oti_t1h = have_1h ? (s->oti - c->oti_hist[0]) : 0.0f;
    const float wti_t1h = have_1h ? (s->wti - c->wti_hist[0]) : 0.0f;

    const uint8_t sev = pdm_severity(s->wti, s->oti, s->oli, v_unb, i_unb, load);
    c->sev_now = sev;

    float *f = c->feat[PDMF_WINDOW - 1];
    f[0]  = s->vl1;  f[1]  = s->vl2;  f[2]  = s->vl3;
    f[3]  = s->il1;  f[4]  = s->il2;  f[5]  = s->il3;
    f[6]  = s->oti;  f[7]  = s->wti;  f[8]  = s->ati;  f[9] = s->oli;
    f[10] = v_avg;   f[11] = i_avg;
    f[12] = (v_unb > 100.0f) ? 100.0f : v_unb;     /* clip as in Python */
    f[13] = (i_unb > 200.0f) ? 200.0f : i_unb;
    f[14] = load;
    f[15] = s->oti - s->ati;
    f[16] = s->wti - s->oti;
    f[17] = oti_roc;
    f[18] = wti_roc;
    f[19] = wti_t1h;
    f[20] = oti_t1h;
    f[21] = (float)sev;
    f[22] = (s->wti - PDMF_WTI_WARN) / 10.0f;

    if (c->count < 0xFFFF) c->count++;
    return (c->count >= PDMF_WINDOW) ? 1 : 0;
}

#ifdef __cplusplus
}
#endif
#endif /* PDM_FEATURES_H */
