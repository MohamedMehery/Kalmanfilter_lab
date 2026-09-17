// Host-side (non-Arduino) build of the self-test, used purely for CI /
// local verification that the exported C model matches the Python
// reference before flashing real hardware. Compiles with plain g++:
//
//   g++ -std=c++17 -I ../model -I ../src test/host_selftest.cpp -o /tmp/host_selftest -lm
//   /tmp/host_selftest
//
// On real AVR hardware the same tfm_inference.h / model headers are used
// unmodified by src/main.cpp via PlatformIO (see platformio.ini).
#include <cstdio>
#include <cstdint>
#include <cmath>

using std::uint8_t;

#include "../src/tfm_inference.h"
#include "test_vectors.h"

int main() {
  printf("=== Transformer Fault MLP -- host self-test ===\n");
  printf("Vectors: %d\n", TFM_N_TEST_VECTORS);

  int correct = 0;
  float max_abs_err = 0.0f;

  for (int v = 0; v < TFM_N_TEST_VECTORS; v++) {
    float x[TFM_N_INPUTS];
    for (int i = 0; i < TFM_N_INPUTS; i++) {
      x[i] = TFM_TEST_X[v * TFM_N_INPUTS + i];
    }
    float p_device = tfm_predict_from_raw(x);
    float p_reference = TFM_TEST_Y_PROBA[v];
    uint8_t y_true = TFM_TEST_Y_LABEL[v];

    uint8_t pred_device = p_device >= 0.5f ? 1 : 0;
    if (pred_device == y_true) correct++;
    float err = fabsf(p_device - p_reference);
    if (err > max_abs_err) max_abs_err = err;

    printf("  [%d] device_p=%.4f  reference_p=%.4f  true_label=%d  pred=%d  |err|=%.6f\n",
           v, p_device, p_reference, y_true, pred_device, err);
  }

  printf("Accuracy on bundled vectors: %d/%d\n", correct, TFM_N_TEST_VECTORS);
  printf("Max |device_p - reference_p|: %.6f\n", max_abs_err);
  printf(max_abs_err < 0.01f
             ? "PASS: device inference matches PC/Colab model.\n"
             : "WARNING: numerical mismatch -- check export step.\n");
  return max_abs_err < 0.01f ? 0 : 1;
}
