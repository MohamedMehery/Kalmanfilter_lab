/****************************************************************************
 * sim_main.cpp — host build of the edge node's feature pipeline
 *
 * Builds the SAME pdm_features.h the ESP32 uses, streams the replay rows
 * through it and prints the resulting windows.  A companion Python script
 * (verify_pipeline.py) feeds those windows to the int8 interpreter and checks
 * the decisions against the host reference in replay_data.h.
 *
 * This lets the whole edge path be validated in CI with no hardware attached.
 *
 *   pio run -e native_sim && .pio/build/native_sim/program
 *   (or simply: gcc/g++ -I include src/sim_main.cpp -o sim)
 ****************************************************************************/
#include <cstdio>
#include <cstdlib>

#include "pdm_features.h"
#include "replay_data.h"

int main(int argc, char **argv)
{
    const float i_rated = (argc > 1) ? (float)atof(argv[1]) : 1449.0f;

    pdm_ctx_t ctx;
    pdm_init(&ctx, i_rated);

    int emitted = 0;
    for (int n = 0; n < REPLAY_N; n++) {
        pdm_sample_t s;
        const float *r = replay_rows[n];
        s.vl1 = r[0]; s.vl2 = r[1]; s.vl3 = r[2];
        s.il1 = r[3]; s.il2 = r[4]; s.il3 = r[5];
        s.oti = r[6]; s.wti = r[7]; s.ati = r[8]; s.oli = r[9];

        if (!pdm_push(&ctx, &s)) continue;

        /* Print the full flattened window: row index, then W*F floats. */
        printf("%d", n);
        for (int t = 0; t < PDMF_WINDOW; t++)
            for (int f = 0; f < PDMF_N_FEATURES; f++)
                printf(" %.6f", ctx.feat[t][f]);
        printf("\n");
        emitted++;
    }
    fprintf(stderr, "[sim] emitted %d windows (sev_now of last = %u)\n",
            emitted, ctx.sev_now);
    return 0;
}
