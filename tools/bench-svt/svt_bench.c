/*
 * svt_bench: ns/call of SVT-AV1's own optimised kernels (its run-time
 * dispatched C / SIMD / asm, as the encoder uses them), on the same inputs
 * as `nc-checkasm --bench`, so the two tables can be compared line by line.
 *
 * Inputs are generated exactly as nc-checkasm's bench does: SplitMix64
 * seeded with 0x5EED fills a source plane, then a reference plane, of
 * STRIDE x ROWS bytes each. Timing is the same too: calibrate to about
 * 20 ms per measurement, report the best of five.
 *
 * Output: one line per kernel, "kernel WxH ns", for tools/bench-svt/compare.py.
 * SVT-AV1 is BSD-3-Clause; this program only calls its public kernel API.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"

#define STRIDE (4 * 64 + 32)
#define ROWS (64 + 8)
#define TARGET_NS 20000000.0
#define REPEATS 5

static uint8_t src[STRIDE * ROWS] __attribute__((aligned(64)));
static uint8_t rf[STRIDE * ROWS] __attribute__((aligned(64)));
static volatile uint32_t sink;

static uint64_t rng_state;
static uint64_t next_u64(void) {
    rng_state += 0x9E3779B97F4A7C15ull;
    uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

typedef void (*bench_fn)(void *ctx);

/* Best-of-REPEATS ns per call of f(ctx). */
static double timed(bench_fn f, void *ctx) {
    uint64_t calls = 16;
    for (;;) {
        double t0 = now_ns();
        for (uint64_t i = 0; i < calls; i++) f(ctx);
        double elapsed = now_ns() - t0;
        if (elapsed >= TARGET_NS / 4 || calls >= (1ull << 30)) {
            calls = (uint64_t)(calls * TARGET_NS / (elapsed > 1 ? elapsed : 1));
            if (calls == 0) calls = 1;
            break;
        }
        calls *= 4;
    }
    double best = 1e300;
    for (int r = 0; r < REPEATS; r++) {
        double t0 = now_ns();
        for (uint64_t i = 0; i < calls; i++) f(ctx);
        double per = (now_ns() - t0) / (double)calls;
        if (per < best) best = per;
    }
    return best;
}

/* ------------------------------------------------------------- kernels */

typedef struct {
    int w, h;
} Size;

static void sad16(void *c) { (void)c; sink += svt_aom_sad16x16(src, STRIDE, rf, STRIDE); }
static void sad64(void *c) { (void)c; sink += svt_aom_sad64x64(src, STRIDE, rf, STRIDE); }

/* AV1 regular filter (libaom sub_pel_filters_8), as SVT's tables. */
static const int16_t regular_8[16][8] = {
    {0, 0, 0, 128, 0, 0, 0, 0},      {0, 2, -6, 126, 8, -2, 0, 0},     {0, 2, -10, 122, 18, -4, 0, 0},
    {0, 2, -12, 116, 28, -8, 2, 0},  {0, 2, -14, 110, 38, -10, 2, 0},  {0, 2, -14, 102, 48, -12, 2, 0},
    {0, 2, -16, 94, 58, -12, 2, 0},  {0, 2, -14, 84, 66, -12, 2, 0},   {0, 2, -14, 76, 76, -14, 2, 0},
    {0, 2, -12, 66, 84, -14, 2, 0},  {0, 2, -12, 58, 94, -16, 2, 0},   {0, 2, -12, 48, 102, -14, 2, 0},
    {0, 2, -10, 38, 110, -14, 2, 0}, {0, 2, -8, 28, 116, -12, 2, 0},   {0, 0, -4, 18, 122, -10, 2, 0},
    {0, 0, -2, 8, 126, -6, 2, 0},
};
static uint8_t conv_out[64 * 64] __attribute__((aligned(64)));

typedef struct {
    int w, h, fx, fy;
} ConvCtx;

static void convolve(void *c) {
    ConvCtx *k = c;
    InterpFilterParams params = {(const int16_t *)regular_8, 8, 16, EIGHTTAP_REGULAR};
    ConvolveParams conv;
    memset(&conv, 0, sizeof(conv));
    conv.round_0 = 3;
    conv.round_1 = 11;
    /* `src` points at the block; the filter reads 3 samples up and left. */
    svt_av1_convolve_2d_sr(rf + 3 * STRIDE + 3, STRIDE, conv_out, k->w, k->w, k->h, &params, &params,
                           k->fx, k->fy, &conv);
    sink += conv_out[0];
}

/* SVT's SIMD assumes 32-byte aligned coefficient buffers. */
#define ALIGNED __attribute__((aligned(64)))

typedef struct {
    int n;
    ALIGNED int16_t residual[32 * 32];
    ALIGNED int32_t coeffs[32 * 32];
    ALIGNED int32_t dequant[32 * 32];
    ALIGNED int32_t qcoeff[32 * 32];
    ALIGNED int32_t dqcoeff[32 * 32];
    ALIGNED uint8_t recon[32 * 32];
} TxCtx;

static void fdct(void *c) {
    TxCtx *k = c;
    if (k->n == 8)
        svt_av1_fwd_txfm2d_8x8(k->residual, k->coeffs, 8, DCT_DCT, 8);
    else
        svt_av1_fwd_txfm2d_32x32(k->residual, k->coeffs, 32, DCT_DCT, 8);
    sink += (uint32_t)k->coeffs[0];
}

static void idct(void *c) {
    TxCtx *k = c;
    TxfmParam param;
    memset(&param, 0, sizeof(param));
    param.tx_type = DCT_DCT;
    param.tx_size = k->n == 8 ? TX_8X8 : TX_32X32;
    param.bd = 8;
    param.eob = k->n * k->n; /* full block: no zero-region shortcut */
    svt_av1_inv_txfm_add(k->dequant, k->recon, k->n, k->recon, k->n, &param);
    sink += k->recon[0];
}

/* libaom-style quantiser tables for qindex steps (dc 120, ac 140). */
static int16_t zbin[2], round_q[2], quant[2], quant_shift[2], dequant_q[2];
static int16_t scan[1024] __attribute__((aligned(64))), iscan[1024] __attribute__((aligned(64)));

static void init_quant(void) {
    const int16_t steps[2] = {120, 140};
    for (int i = 0; i < 2; i++) {
        const int q = steps[i];
        /* libaom av1_build_quantizer: invert_quant(&quant, &shift, q). */
        unsigned t = q;
        int l = 0;
        while (t > 1) { t >>= 1; l++; }
        const int m = 1 + (1 << (16 + l)) / q;
        quant[i] = (int16_t)(m - (1 << 16));
        quant_shift[i] = (int16_t)(1 << (16 - l));
        zbin[i] = (int16_t)((q * 64 + 64) >> 7);
        round_q[i] = (int16_t)((q * 48) >> 7);
        dequant_q[i] = (int16_t)q;
    }
    for (int i = 0; i < 1024; i++) scan[i] = iscan[i] = (int16_t)i;
}

static void quantize32(void *c) {
    TxCtx *k = c;
    uint16_t eob;
    svt_aom_quantize_b(k->coeffs, 1024, zbin, round_q, quant, quant_shift, k->qcoeff, k->dqcoeff,
                       dequant_q, &eob, scan, iscan, NULL, NULL, 1);
    sink += eob;
}

static void subpel16(void *c) {
    (void)c;
    uint32_t sse;
    /* SVT filters its first argument: the reference. Offsets 3/8, 5/8. */
    sink += svt_aom_sub_pixel_variance16x16(rf, STRIDE, 3, 5, src, STRIDE, &sse);
}

int main(void) {
    rng_state = 0x5EED;
    for (int i = 0; i < STRIDE * ROWS; i++) src[i] = (uint8_t)next_u64();
    for (int i = 0; i < STRIDE * ROWS; i++) rf[i] = (uint8_t)next_u64();

    EbCpuFlags flags = svt_aom_get_cpu_flags_to_use();
    svt_aom_setup_common_rtcd_internal(flags);
    svt_aom_setup_rtcd_internal(flags);
    init_quant();

    printf("sad 16x16 %.2f\n", timed(sad16, NULL));
    printf("sad 64x64 %.2f\n", timed(sad64, NULL));

    ConvCtx c16 = {16, 16, 5, 9}, c64 = {64, 64, 5, 9};
    printf("av1_predict_2d 16x16 %.2f\n", timed(convolve, &c16));
    printf("av1_predict_2d 64x64 %.2f\n", timed(convolve, &c64));

    static TxCtx tx;
    for (int n = 8; n <= 32; n += 24) {
        tx.n = n;
        for (int i = 0; i < n * n; i++) tx.residual[i] = (int16_t)(src[i] - 128);
        printf("av1_forward_dct %dx%d %.2f\n", n, n, timed(fdct, &tx));
        /* Dequantised coefficients as nc-checkasm's bench produces them. */
        for (int i = 0; i < n * n; i++) {
            const int step = i == 0 ? 120 : 140;
            const int mag = (abs(tx.coeffs[i]) * 128 + step * 48) / (step * 128);
            tx.dequant[i] = (tx.coeffs[i] < 0 ? -mag : mag) * step;
        }
        memset(tx.recon, 128, sizeof(tx.recon));
        printf("av1_inverse_dct_add %dx%d %.2f\n", n, n, timed(idct, &tx));
    }
    printf("av1_quantize 32x32 %.2f\n", timed(quantize32, &tx));
    printf("sub_pixel_variance 16x16 %.2f\n", timed(subpel16, NULL));
    return 0;
}
