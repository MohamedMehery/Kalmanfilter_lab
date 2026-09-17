/****************************************************************************
 * Tiny inference engine for the transformer predictive-maintenance MLP.
 *
 * Pure C/C++, no external ML libraries needed — mirrors the "TinyAI" style
 * hand-rolled MLP forward-pass already used in this repo's
 * "Edge AI Driver-Behavior & Vehicle-Health Classification" project, but
 * generalized to read weights that may live in PROGMEM (AVR) or plain
 * flash-mapped RAM (ESP32 / STM32).
 *
 * Model: Dense(N_IN -> 16, ReLU) -> Dense(16 -> 8, ReLU) -> Dense(8 -> 1, Sigmoid)
 ****************************************************************************/
#pragma once

#include <math.h>
#include "transformer_fault_model.h"

static inline float tfm_relu(float x) { return x > 0.0f ? x : 0.0f; }
static inline float tfm_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Standardize raw features in-place using the training-set mean/scale
// (StandardScaler: z = (x - mean) / scale).
static inline void tfm_standardize(float *features, int n) {
  for (int i = 0; i < n; i++) {
    float mean = TFM_READ_FLOAT(&TFM_SCALER_MEAN[i]);
    float scale = TFM_READ_FLOAT(&TFM_SCALER_SCALE[i]);
    features[i] = (features[i] - mean) / scale;
  }
}

// Runs the full forward pass on an already-standardized feature vector of
// length TFM_N_INPUTS. Returns the fault probability in [0, 1].
static inline float tfm_predict_proba(const float *x_std) {
  float h1[TFM_N_HIDDEN1];
  for (int o = 0; o < TFM_N_HIDDEN1; o++) {
    float acc = TFM_READ_FLOAT(&TFM_B1[o]);
    for (int i = 0; i < TFM_N_INPUTS; i++) {
      acc += x_std[i] * TFM_READ_FLOAT(&TFM_W1[o * TFM_N_INPUTS + i]);
    }
    h1[o] = tfm_relu(acc);
  }

  float h2[TFM_N_HIDDEN2];
  for (int o = 0; o < TFM_N_HIDDEN2; o++) {
    float acc = TFM_READ_FLOAT(&TFM_B2[o]);
    for (int i = 0; i < TFM_N_HIDDEN1; i++) {
      acc += h1[i] * TFM_READ_FLOAT(&TFM_W2[o * TFM_N_HIDDEN1 + i]);
    }
    h2[o] = tfm_relu(acc);
  }

  float acc_out = TFM_READ_FLOAT(&TFM_B3[0]);
  for (int i = 0; i < TFM_N_HIDDEN2; i++) {
    acc_out += h2[i] * TFM_READ_FLOAT(&TFM_W3[i]);
  }
  return tfm_sigmoid(acc_out);
}

// Convenience wrapper: copies raw features, standardizes, and predicts.
// `raw_features` must have exactly TFM_N_INPUTS entries in the feature
// order documented at the top of transformer_fault_model.h.
static inline float tfm_predict_from_raw(const float *raw_features) {
  float buf[TFM_N_INPUTS];
  for (int i = 0; i < TFM_N_INPUTS; i++) buf[i] = raw_features[i];
  tfm_standardize(buf, TFM_N_INPUTS);
  return tfm_predict_proba(buf);
}
