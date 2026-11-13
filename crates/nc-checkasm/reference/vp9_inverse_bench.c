//! Matched-fixture libvpx inverse-transform timing; no implementation copied.
// Build with libvpx source/build includes and its ISA objects or static library.
// libvpx adds and clips to u8; nc-dsp returns unclipped i32 residuals. Report
// this API difference rather than presenting the two timings as identical work.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "vpx_dsp/txfm_common.h"

#ifdef __aarch64__
#define IHT8 vp9_iht8x8_64_add_neon
#define IHT16 vp9_iht16x16_256_add_neon
#define IDCT32 vpx_idct32x32_1024_add_neon
#else
#define IHT8 vp9_iht8x8_64_add_sse2
#define IHT16 vp9_iht16x16_256_add_sse2
#define IDCT32 vpx_idct32x32_1024_add_avx2
#endif
void IHT8(const tran_low_t *, uint8_t *, int, int);
void IHT16(const tran_low_t *, uint8_t *, int, int);
void IDCT32(const tran_low_t *, uint8_t *, int);

typedef void (*Kernel)(const tran_low_t *, uint8_t *, int, int);
static void idct32(const tran_low_t *in, uint8_t *out, int stride, int kind) {
    (void)kind;
    IDCT32(in, out, stride);
}
static uint64_t random64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
int main(void) {
    enum { MAX_SAMPLES = 32 * 32, CALLS = 10000, REPEATS = 7 };
    _Alignas(32) tran_low_t input[MAX_SAMPLES];
    _Alignas(32) uint8_t output[MAX_SAMPLES];
    uint64_t state = 0x5eed;
    for (int i = 0; i < MAX_SAMPLES; ++i)
        input[i] = ((int)(uint8_t)random64(&state) - 128) * 8;
    // nc-checkasm orders column,row as DD,DA,AD,AA; libvpx uses DD,AD,DA,AA.
    const int kinds[] = {0, 2, 1, 3};
    const char *names[] = {"dd", "da", "ad", "aa"};
    volatile unsigned checksum = 0;
    for (int side = 8; side <= 32; side *= 2) {
        Kernel kernel = side == 8 ? IHT8 : side == 16 ? IHT16 : idct32;
        for (int kind = 0; kind < (side == 32 ? 1 : 4); ++kind) {
            double best = 1e30;
            memset(output, 128, sizeof(output));
            for (int repeat = 0; repeat < REPEATS; ++repeat) {
                double start = now();
                for (int call = 0; call < CALLS; ++call)
                    kernel(input, output, side, kinds[kind]);
                double ns = (now() - start) / CALLS;
                if (ns < best) best = ns;
                checksum += output[0];
            }
            printf("libvpx_inverse_%s %dx%d %.2f ns\n", names[kind], side, side, best);
        }
    }
    fprintf(stderr, "checksum=%u\n", checksum);
    return 0;
}
